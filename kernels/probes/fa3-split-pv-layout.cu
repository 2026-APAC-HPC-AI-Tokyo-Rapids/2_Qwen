// Exhaustive CPU check of the RS-register -> SS-shared probability mailbox.
// This does not substitute for GPU byte checks or numerical attention checks.
#define APAC_SPLIT_PV_LAYOUT_ONLY
#include "fa3-split-pv.cu"
#include <cstdio>
#include <array>
#include <vector>
#include <cstring>

int main() {
    using namespace cute;
    using namespace apac_split_pv;
    std::vector<int> owners(128 * 192, 0), image(128 * 192, -1);
    int mismatches = 0, row_mismatches = 0, noncontiguous_words = 0;
    for (int tid = 0; tid < 256; ++tid) {
        ML::TiledMmaQK qk;
        auto coords = qk.get_thread_slice(tid).partition_C(make_identity_tensor(select<0, 1>(ShapeQK{})));
        auto scores = partition_fragment_C(qk, select<0, 1>(ShapeQK{}));
        for (int i = 0; i < size(scores); ++i) { scores(i) = get<0>(coords(i)) * 192 + get<1>(coords(i)); }
        // Exact host counterpart of the deployed permute_Cregs_fp8: swap
        // pairs of FP32 slots; values remain exact integer coordinate IDs.
        auto pairs = group_modes<1, 3>(recast<uint2>(scores));
        for (int m = 0; m < size<1>(pairs); ++m) {
            for (int i = 0; i < size<0, 2>(pairs) / 2; ++i) {
                // memcpy avoids host strict-aliasing UB from uint2 over float.
                auto* pa = &pairs(make_coord(_0{}, _1{}, 2*i), m);
                auto* pb = &pairs(make_coord(_0{}, _0{}, 2*i+1), m);
                uint2 a, b;
                std::memcpy(&a, pa, sizeof(a));
                std::memcpy(&b, pb, sizeof(b));
                std::memcpy(pa, &b, sizeof(b));
                std::memcpy(pb, &a, sizeof(a));
            }
        }
        // Host expression of the pinned utility's SM90 FP8 layout branch.
        auto layout = scores.layout();
        auto split = logical_divide(get<0, 2>(layout), Tile<Layout<Shape<_2, _2>>>{});
        auto rs_layout = make_layout(make_layout(Layout<_4>{}, get<0, 0, 0>(split), get<0, 0, 1>(split)),
            get<1>(layout), coalesce(make_layout(get<0, 1>(split), get<2>(layout))));
        auto packed = make_tensor(scores.data(), rs_layout);
        auto dest = ML::TiledMmaPV{}.get_thread_slice(tid).partition_A(make_identity_tensor(Shape<_128, _192>{}));
        auto physical_layout = make_composed_layout(PLayout{}.layout_a(), _0{}, PLayout{}.layout_b());
        auto offsets = ML::TiledMmaPV{}.get_thread_slice(tid).partition_A(make_tensor(counting_iterator<int>{}, physical_layout)(_, _, _0{}));
        for (int i = 0; i < size(offsets); i += 4) {
            bool contiguous = offsets(i) % 4 == 0;
            for (int j = 1; j < 4; ++j) { contiguous &= offsets(i+j) == offsets(i)+j; }
            noncontiguous_words += !contiguous;
        }
        static_assert(decltype(size(packed))::value == decltype(size(dest))::value);
        for (int i = 0; i < size(packed); ++i) {
            int index = get<0>(dest(i))*192 + get<1>(dest(i));
            ++owners.at(index);
            image.at(index) = int(packed(i));
            mismatches += int(packed(i)) != index;
            row_mismatches += int(packed(i))/192 != index/192;
        }
    }
    int missing_or_duplicate = 0;
    for (int i = 0; i < 128*192; ++i) { missing_or_duplicate += owners[i] != 1; }
    // The deployed V transpose deliberately permutes K coordinates as well as
    // output columns. Identity against logical softmax P is therefore the wrong
    // oracle. Reconstruct V independently from the real LDSM.T / PRMT / STSM.N
    // instruction bit layouts and check that the paired K indices agree.
    constexpr int bytes = 192*128;
    std::array<int, bytes> v_inverse;
    v_inverse.fill(-1);
    auto src = make_tensor(counting_iterator<int>{}, make_layout(Shape<_128,_192,_2>{}, Stride<_1,_128,Int<bytes>>{}));
    auto vl = make_composed_layout(ML::SmemLayoutVtMma{}.layout_a(), _0{}, ML::SmemLayoutVtMma{}.layout_b());
    auto dst = make_tensor(counting_iterator<int>{}, vl);
    using Load = Copy_Traits<SM75_U16x8_LDSM_T>;
    using Store = Copy_Traits<SM90_U32x4_STSM_N>;
    int source[128][32], dest[128][32], stage_errors = 0;
    for (int stage=0; stage<2; ++stage) {
        for (int i=0; i<bytes/(128*32); ++i) {
            for (int thread=0; thread<128; ++thread) {
                auto a = ML::S2RTiledCopyVt{}.get_thread_slice(thread).partition_S(flat_divide(src, ML::LDSM_divide_shape{}));
                auto b = ML::R2STiledCopyV{}.get_thread_slice(thread).partition_D(flat_divide(dst, ML::STSM_divide_shape{}));
                auto aa = logical_divide(group_modes<1,rank(a)-1>(a), Shape<Underscore,_2>{});
                auto bb = logical_divide(group_modes<1,rank(b)-1>(b), Shape<Underscore,_2>{});
                for (int j=0; j<32; ++j) {
                    source[thread][j] = aa(j%16,make_coord(j/16,i),stage)-stage*bytes;
                    dest[thread][j] = bb(j%16,make_coord(j/16,i),stage)-stage*bytes;
                }
            }
            constexpr int perm[8] = {0,2,4,6,1,3,5,7};
            for (int thread=0; thread<128; ++thread) for (int j=0; j<32; ++j) {
                int group=j/16, out_byte=j%16;
                int in_byte=(out_byte/8)*8+perm[out_byte%8];
                int lbit=Load::DstLayout{}(make_coord(thread%32,in_byte*8));
                int from=source[thread/32*32+lbit/128][group*16+(lbit%128)/8];
                int sbit=Store::SrcLayout{}(make_coord(thread%32,out_byte*8));
                int to=dest[thread/32*32+sbit/128][group*16+(sbit%128)/8];
                if (stage) { stage_errors += v_inverse.at(to) != from; }
                else {
                    if (v_inverse.at(to) != -1) { return 3; }
                    v_inverse.at(to) = from;
                }
            }
        }
    }
    int pairing_errors = 0, v_column_errors = 0;
    for (int k=0; k<192; ++k) for (int d=0; d<128; ++d) {
        int original_v = v_inverse.at(vl(make_coord(d,k,0)));
        v_column_errors += original_v%128 != v_inverse.at(vl(make_coord(d,0,0)))%128;
        for (int row=0; row<128; ++row) {
            pairing_errors += image.at(row*192+k)%192 != original_v/128;
        }
    }
    std::vector<int> out_owners(128*128, 0);
    int output_errors = 0;
    for (int tid=0; tid<128; ++tid) {
        auto acc = partition_fragment_C(PV{}, select<0,1>(ShapePV{}));
        auto coordinates = PV{}.get_thread_slice(tid).partition_C(make_identity_tensor(select<0,1>(ShapePV{})));
        for (int i=0; i<size(acc); ++i) {
            int d=get<1>(coordinates(i));
            acc(i)=get<0>(coordinates(i))*128+v_inverse.at(vl(make_coord(d,0,0)))%128;
        }
        auto values=group_modes<1,3>(acc);
        for (int m=0; m<size<1>(values); ++m) for (int r=0; r<2; ++r) for (int n=0; n<8; ++n) {
            std::swap(values(make_coord(1,r,2*n),m),values(make_coord(0,r,2*n+1),m));
        }
        auto coords=group_modes<1,3>(coordinates);
        for (int wave=0; wave<size<1>(values); ++wave) for (int r=0; r<2; ++r) {
            int row=get<0>(coords(make_coord(0,r,0),wave));
            int col=2*get<1>(coords(_0{},wave));
            for (int n=0; n<8; ++n) for (int j=0; j<4; ++j) {
                int index=row*128+col+16*n+j;
                ++out_owners.at(index);
                output_errors += int(values(make_coord(j%2,r,2*n+j/2),wave)) != index;
            }
        }
    }
    for (int n: out_owners) { output_errors += n != 1; }
    std::printf("{\"shared_bytes\":%zu,\"matrix_elements\":24576,\"native_probability_permuted_elements\":%d,\"row_errors\":%d,\"ownership_errors\":%d,\"pv_key_pairing_errors\":%d,\"v_column_errors\":%d,\"stage_errors\":%d,\"noncontiguous_words\":%d,\"output_errors\":%d,\"gpu_executed\":false}\n", sizeof(Shared), mismatches, row_mismatches, missing_or_duplicate, pairing_errors, v_column_errors, stage_errors, noncontiguous_words, output_errors);
    return row_mismatches || missing_or_duplicate || pairing_errors || v_column_errors || stage_errors || noncontiguous_words || output_errors;
}
