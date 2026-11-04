// CPU-only execution of the actual patched loop. WGMMA and TMA are logical
// stubs; this checks all K blocks, scale indices, stage phases and releases.
#include <cassert>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#define CUTLASS_LAMBDA_FUNC_INLINE
namespace cute {
template<int N> using Int = std::integral_constant<int,N>;
int block_rank_in_cluster() {return 0;}
}
float sfa_data[3][256], sfb_data[64];
int logical[3], next_k, releases, mma64_calls, mma128_calls;
bool available[3];
float scale_a(int k,int row) {return (k+1)*.001f+(row+1)*.0001f;}
float scale_b(int k) {return (k+1)*.003f+.01f;}
float value(int k,int wave,int half,int reg,int kk) {return float((k+1)*8+wave*4+half*2+reg+kk);}
struct Desc {uint32_t reg32_[2]={0,0};};
struct Full {
 int stage;
 void wait(uint32_t phase) {
  assert(stage==next_k%3 && phase==uint32_t((next_k/3)%2));
  assert(available[stage]); available[stage]=false; logical[stage]=next_k++;
  for(int r=0;r<256;++r) sfa_data[stage][r]=scale_a(logical[stage],r);
 }
};
struct Empty {
 int stage;
 void arrive(int) {assert(!available[stage]); available[stage]=true; ++releases;}
};
struct Fake64 {
 static void wgmma(const Desc& a,const Desc& b,float* dst,bool add) {
  int stage=a.reg32_[0]/2048, wave=(a.reg32_[0]%2048)*16/128/128;
  int half=(b.reg32_[0]%1024)*16/128/64, kk=(a.reg32_[0]*16%128)/32;
  assert(stage==int(b.reg32_[0]/1024) && !available[stage]);
  ++mma64_calls;
  for(int i=0;i<32;++i) {float v=value(logical[stage],wave,half,i,kk);dst[i]=add?dst[i]+v:v;}
 }
};
struct Fake128 {
 static void wgmma(const Desc& a,const Desc& b,float* dst,bool add) {
  int stage=a.reg32_[0]/2048,wave=(a.reg32_[0]%2048)*16/128/128,kk=(a.reg32_[0]*16%128)/32;
  assert(stage==int(b.reg32_[0]/1024) && !available[stage]);
  ++mma128_calls;
  for(int i=0;i<64;++i) {float v=value(logical[stage],wave,i/32,i%32,kk);dst[i]=add?dst[i]+v:v;}
 }
};
namespace mma::sm90 {template<int N> struct FP8MMASelector {static_assert(N==64);using type=Fake64;};}
namespace ptx {
void warpgroup_fence_operand(float&) {}
void warpgroup_arrive() {}
void warpgroup_commit_batch() {}
template<int N> void warpgroup_wait() {static_assert(N==0||N==1);}
float ld_shared(const float* p) {assert(p>=sfb_data && p<sfb_data+64 || p>=&sfa_data[0][0] && p<&sfa_data[2][256]);return *p;}
}
float __fmaf_rn(float a,float b,float c) {return std::fma(a,b,c);}
int main() {
 for(uint32_t num_total_k_blocks: {32u,64u}) {
  next_k=releases=mma64_calls=mma128_calls=0;
  for(int i=0;i<3;++i) {available[i]=true;logical[i]=-1;}
  for(int i=0;i<64;++i) sfb_data[i]=scale_b(i);
  constexpr uint32_t BLOCK_K=128,kNumStages=3,kNumTMAMulticast=2,SMEM_A_SIZE_PER_STAGE=32768,SMEM_B_SIZE_PER_STAGE=16384,WAVE_BLOCK_M=128;
  uint32_t stage_idx=0,phase=0,lane_idx=0,r_0=0,r_1=8,a_desc_lo=0,b_desc_lo=0;
  Full full[3]={{0},{1},{2}};Full* full_barriers[3]={full,full+1,full+2};
  Empty empty[3]={{0},{1},{2}};Empty* empty_barriers[3]={empty,empty+1,empty+2};
  float* smem_sfa[3]={sfa_data[0],sfa_data[1],sfa_data[2]};float* smem_sfb=sfb_data;
  struct {bool is_peer_cta_alive=true;} scheduler;
  using WGMMA=Fake128;
  Desc a_desc,b_desc;
  float final_accum[128]={0};
// ADVANCE_PIPELINE
// R44_BODY
  assert(next_k==int(num_total_k_blocks) && releases==int(num_total_k_blocks));
  for(bool free:available) assert(free);
  for(int wave=0;wave<2;++wave) for(int half=0;half<2;++half) for(int i=0;i<32;++i) {
   float expected=0;
   for(int k=0;k<int(num_total_k_blocks);++k) {
#if DG_R44_DIAGNOSTIC_NO_PROMOTION
    for(int kk=0;kk<4;++kk) expected+=value(k,wave,half,i,kk);
#else
    float product=0;for(int kk=0;kk<4;++kk) product+=value(k,wave,half,i,kk);
    int row=wave*128+(i%4>=2?8:0);
    expected=std::fma(scale_a(k,row)*scale_b(k),product,expected);
#endif
   }
   assert(std::bit_cast<uint32_t>(final_accum[wave*64+half*32+i])==std::bit_cast<uint32_t>(expected));
  }
#if DG_R44_DIAGNOSTIC_NO_PROMOTION
  assert(mma128_calls==int(num_total_k_blocks)*8 && mma64_calls==0);
#else
  assert(mma64_calls==int(num_total_k_blocks)*16 && mma128_calls==0);
#endif
  std::cout << "K128 blocks=" << next_k << " releases=" << releases << " mma64=" << mma64_calls << " mma128=" << mma128_calls << " all 128 fragment outputs bitwise equal\n";
 }
}
