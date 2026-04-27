from models.psp.utils.psp_utils import get_knn_pts, index_points
import torch.nn as nn
import torch
from einops import repeat



def build_bn_1d(num_channels, use_bn):
    return nn.BatchNorm1d(num_channels) if use_bn else nn.Identity()


def build_bn_2d(num_channels, use_bn):
    return nn.BatchNorm2d(num_channels) if use_bn else nn.Identity()


def square_distance(src, dst):
    """
    src: [B, 3, N]
    dst: [B, 3, M]
    return: [B, N, M]
    """
    src_t = src.transpose(1, 2).contiguous()
    dst_t = dst.transpose(1, 2).contiguous()
    return torch.cdist(src_t, dst_t, p=2) ** 2


def gather_points(points, idx):
    """
    points: [B, C, N]
    idx: [B, S]
    return: [B, C, S]
    """
    B, C, _ = points.shape
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(points, 2, idx_expand)


def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, 3, N]
    return: fps_idx [B, npoint]
    """
    device = xyz.device
    B, _, N = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    xyz_t = xyz.transpose(1, 2).contiguous()  # [B, N, 3]

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_t[batch_indices, farthest, :].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz_t - centroid) ** 2, dim=-1)          # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def three_nn_interpolate(src_xyz, src_feat, dst_xyz, eps=1e-8):
    """
    src_xyz:  [B, 3, S]
    src_feat: [B, C, S]
    dst_xyz:  [B, 3, N]
    return:   [B, C, N]
    """
    dist = square_distance(dst_xyz, src_xyz)  # [B, N, S]
    k = min(3, dist.shape[-1])

    dists, idx = torch.topk(
        dist, k=k, dim=-1, largest=False, sorted=False
    )  # [B, N, k]

    inv_dists = 1.0 / (dists + eps)
    norm = torch.sum(inv_dists, dim=-1, keepdim=True)
    weight = inv_dists / norm  # [B, N, k]

    nn_feat = index_points(src_feat, idx)     # [B, C, N, k]
    out = torch.sum(nn_feat * weight.unsqueeze(1), dim=-1)  # [B, C, N]
    return out


def group_feature_by_knn(query_xyz, support_xyz, support_feat, k):
    """
    query_xyz:    [B, 3, Nq]
    support_xyz:  [B, 3, Ns]
    support_feat: [B, C, Ns]

    return:
        grouped_feat: [B, C, Nq, K]
        grouped_xyz:  [B, 3, Nq, K]
        knn_idx:      [B, Nq, K]
    """
    _, knn_idx = get_knn_pts(k, support_xyz, query_xyz, return_idx=True)
    grouped_feat = index_points(support_feat, knn_idx)
    grouped_xyz = index_points(support_xyz, knn_idx)
    return grouped_feat, grouped_xyz, knn_idx


class EncoderAttentionCtx(nn.Module):
    """
    给 context 分支用的通用 attention，不改原来的 Encoder_Attention。
    """
    def __init__(self, channels, k, geo_dim=3, use_bn=True):
        super().__init__()
        self.k = k
        self.channels = channels

        self.q_conv = nn.Conv1d(channels, channels, 1)
        self.k_conv = nn.Conv1d(channels, channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.geo_mlp = nn.Sequential(
            nn.Conv2d(geo_dim, channels // 2, 1),
            build_bn_2d(channels // 2, use_bn),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
        )

        self.rel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            build_bn_2d(channels // 2, use_bn),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            build_bn_1d(channels * 2, use_bn),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2, channels, 1),
            build_bn_1d(channels, use_bn),
        )

    def forward(self, pts, feats, geos=None, knn_idx=None):
        if geos is None:
            geos = pts

        q = self.q_conv(feats)
        k = self.k_conv(feats)
        v = self.v_conv(feats)

        if knn_idx is None:
            _, knn_idx = get_knn_pts(min(self.k, pts.shape[-1]), pts, pts, return_idx=True)

        K = knn_idx.shape[-1]

        knn_geos = index_points(geos, knn_idx)
        geo_embedding = self.geo_mlp(geos.unsqueeze(-1) - knn_geos)

        q_expand = repeat(q, 'b c n -> b c n k', k=K)
        knn_k = index_points(k, knn_idx)
        knn_v = index_points(v, knn_idx)

        attn = torch.softmax(self.rel_mlp(q_expand - knn_k + geo_embedding), dim=-1)
        agg_feat = torch.einsum('bcnk,bcnk->bcn', attn, knn_v + geo_embedding) + feats
        out_feat = self.out_conv(agg_feat) + agg_feat
        return out_feat


class DenseLocalGeometryStageCtx(nn.Module):
    def __init__(self, in_channels, growth_channels, out_channels, num_blocks, k, geo_dim=3, use_bn=True):
        super().__init__()
        self.num_blocks = num_blocks

        self.compress_mlps = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        current_in = in_channels
        for _ in range(num_blocks):
            self.compress_mlps.append(
                nn.Sequential(
                    nn.Conv1d(current_in, growth_channels, 1),
                    build_bn_1d(growth_channels, use_bn),
                    nn.ReLU(inplace=True),
                )
            )
            self.attn_blocks.append(
                EncoderAttentionCtx(
                    channels=growth_channels,
                    k=k,
                    geo_dim=geo_dim,
                    use_bn=use_bn,
                )
            )
            current_in += growth_channels

        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels + num_blocks * growth_channels, out_channels, 1),
            build_bn_1d(out_channels, use_bn),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts, feat, geos=None, knn_idx=None):
        feats_list = [feat]

        for compress, attn in zip(self.compress_mlps, self.attn_blocks):
            dense_feat = torch.cat(feats_list, dim=1)
            x = compress(dense_feat)
            x = attn(pts, x, geos=geos, knn_idx=knn_idx)
            feats_list.append(x)

        out = self.fuse(torch.cat(feats_list, dim=1))
        return out


class TransitionDownCtx(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, use_bn=True):
        super().__init__()
        self.ratio = ratio
        self.k = k

        self.local_mlp = nn.Sequential(
            nn.Conv2d(in_channels + 3, out_channels, 1),
            build_bn_2d(out_channels, use_bn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            build_bn_2d(out_channels, use_bn),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, feat):
        """
        xyz:  [B, 3, N]
        feat: [B, C, N]
        """
        B, _, N = xyz.shape
        S = max(8, int(N * self.ratio))
        S = min(S, N)

        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = gather_points(xyz, fps_idx)

        grouped_feat, grouped_xyz, _ = group_feature_by_knn(
            query_xyz=new_xyz,
            support_xyz=xyz,
            support_feat=feat,
            k=min(self.k, N),
        )

        rel_xyz = new_xyz.unsqueeze(-1) - grouped_xyz
        grouped_input = torch.cat([grouped_feat, rel_xyz], dim=1)

        new_feat = self.local_mlp(grouped_input).max(dim=-1)[0]
        return new_xyz, new_feat


class TransitionUpCtx(nn.Module):
    def __init__(self, low_channels, skip_channels, out_channels, use_bn=True):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv1d(low_channels, out_channels, 1),
            build_bn_1d(out_channels, use_bn),
            nn.ReLU(inplace=True),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, out_channels, 1),
            build_bn_1d(out_channels, use_bn),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1),
            build_bn_1d(out_channels, use_bn),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_xyz, low_feat, skip_xyz, skip_feat):
        up_feat = three_nn_interpolate(low_xyz, self.low_proj(low_feat), skip_xyz)
        skip_feat = self.skip_proj(skip_feat)
        out = self.fuse(torch.cat([up_feat, skip_feat], dim=1))
        return out


class HierarchicalContextPropagationCtx(nn.Module):
    def __init__(
        self,
        in_channels,
        ctx_channels,
        ratio1=0.25,
        ratio2=0.25,
        k_down=16,
        k_ctx=16,
        use_bn=True,
    ):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, ctx_channels, 1),
            build_bn_1d(ctx_channels, use_bn),
            nn.ReLU(inplace=True),
        )

        self.down1 = TransitionDownCtx(
            in_channels=ctx_channels,
            out_channels=ctx_channels,
            ratio=ratio1,
            k=k_down,
            use_bn=use_bn,
        )
        self.down2 = TransitionDownCtx(
            in_channels=ctx_channels,
            out_channels=ctx_channels,
            ratio=ratio2,
            k=k_down,
            use_bn=use_bn,
        )

        self.ctx_stage1 = DenseLocalGeometryStageCtx(
            in_channels=ctx_channels,
            growth_channels=ctx_channels,
            out_channels=ctx_channels,
            num_blocks=2,
            k=k_ctx,
            geo_dim=3,
            use_bn=use_bn,
        )
        self.ctx_stage2 = DenseLocalGeometryStageCtx(
            in_channels=ctx_channels,
            growth_channels=ctx_channels,
            out_channels=ctx_channels,
            num_blocks=2,
            k=k_ctx,
            geo_dim=3,
            use_bn=use_bn,
        )

        self.up21 = TransitionUpCtx(
            low_channels=ctx_channels,
            skip_channels=ctx_channels,
            out_channels=ctx_channels,
            use_bn=use_bn,
        )
        self.up10 = TransitionUpCtx(
            low_channels=ctx_channels,
            skip_channels=ctx_channels,
            out_channels=ctx_channels,
            use_bn=use_bn,
        )

        self.post = nn.Sequential(
            nn.Conv1d(ctx_channels, ctx_channels, 1),
            build_bn_1d(ctx_channels, use_bn),
            nn.ReLU(inplace=True),
        )

    def forward(self, p0, f0):
        """
        p0: [B, 3, N]
        f0: [B, C, N]
        """
        f0 = self.pre(f0)

        p1, f1 = self.down1(p0, f0)
        _, knn1 = get_knn_pts(min(16, p1.shape[-1]), p1, p1, return_idx=True)
        f1 = self.ctx_stage1(p1, f1, knn_idx=knn1)

        p2, f2 = self.down2(p1, f1)
        _, knn2 = get_knn_pts(min(16, p2.shape[-1]), p2, p2, return_idx=True)
        f2 = self.ctx_stage2(p2, f2, knn_idx=knn2)

        f1_up = self.up21(p2, f2, p1, f1)
        f0_up = self.up10(p1, f1_up, p0, f0)

        return self.post(f0_up)