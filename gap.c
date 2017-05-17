// Search for large primegaps, written by Robert Gerbicz
// version 0.03

// my long compilation line: gcc -flto -m64 -fopenmp -O2 -fomit-frame-pointer -mavx2 -mtune=skylake -march=skylake -o gap gap3.c -lm
// don't forget -fopenmp  [for OpenMP]
// use your own processor type, mine is skylake

// For newer/older processors use your own better settings for the below constants

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "omp.h" // for multithreading, need gcc >= 4.2

#define version "0.03"

typedef unsigned int ui32;
typedef signed int si32;
typedef long long int si64;
typedef unsigned long long int ui64;


// these are only used to show as a hint, code will ask these as input/switch/file,
// so there is no need to modify these here.

#define default_sieve_bits_log2  25 // Total number of sieve array's bits(!!!), the most important constant
                            // say you have 3 MB of L3 cache, then sb=24 is a good choice since 2^24 bits=2 MB
                            // 2nd example:
                            // for 8 MB Intel's smart cache size use sb=25 (it is also the default here)
                            //    though it could seem a suboptimal choice, because 2^26 bits is exactly 8 MB
                            //    but test runs confirms that sb=25 is better than sb=26
                            // you can see this as "sb" in the code

#define default_bucket_size_log2 14 // in bytes, you can see this as "bs" in the code

#define default_mingap 1346 // searching for gap>=mingap
                            // currently the smallest unknown case is gap=1346
                            // see http://www.trnicely.net/gaps/gaplist.html

#define default_gap_delta 196 // we will sieve mingap2*k+[0,gap_delta) intervals

// *******************
// modify these 6 constants here, we won't input this:
#define save_nextprimetest 1 // we use another big array to save one next_prime() cost after the prec_prime()
                             // in most of the cases. note that if you would use save_nextprimetest=0
                             //  to save memory (but get slower sieve!)
                             //  then the optimal gap_delta should be larger by roughly 10,
                             //  so in the high n range it should be 196+10=206 ( so close to 200 )
                             //  [the hint about default_gap_delta is showing the delta for save_nextprimetest=1]

#define default_report_gap 1000 // we print and save gap=p2-p1 iff gap>=default_report_gap or gap>=mingap
                                // though it is NOT an exhustive search(!!!), only for >=mingap

#define count_LEN_intervals 32 // to lower some init costs in the sieve, 32 is quite a good choice
                               //  and changing to a much larger value would need a much longer
                               //  sieve array, so a much larger memory
                               // note: this can be non-power of two also.

#define MAX_SIZE (1LL<<29) // in bytes for offsets array, to save some space (0.5-1 GB) lower this 
                           // to say (1LL<<24), but that will give a slower sieve
                           // this can be also non-power of two

#define MAX_NUM_SOLUTIONS 1024 // max. number of solutions per thread, we still can print/save results if
                             // there would be more solutions

#define ALIGNEMENT 4096 // alignement (in bytes!)
                        // it could be say (1<<bucket_size_log2), and at least 64.
                        // or a higher power of two, but that gives no speedup
                        // [it should be a power of two]

// *******************

double used_memory; // in gigabytes
double max_memory_gigabytes;// we'll set this, here 1 Gbyte=2^30 bytes=1073741824 bytes
                            // the code try to use no more memory than this.

int sieve_bits_log2; // Use the -sb switch to give it in the code
int bucket_size_log2;// Use the -bs switch
int mingap;          // Use the -gap switch
int mingap2;
int gap_delta;       // Use the -delta switch

#define inf64 0xffffffffffffffff // 2^64-1
#define inf63 0x7fffffffffffffff // 2^63-1
#define inf32 0xffffffff         // 2^32-1
#define inf31 0x7fffffff         // 2^31-1
#define DP30  1073741824.0       // 2.0^30
#define MP64  0xffffffffffffffc5 // precprime(2^64)=2^64-59
#define size_ui32 (sizeof(ui32)) // at least 4
#define size_ui64 (sizeof(ui64)) // it should be 8


#define get_lsb(a) (__builtin_ffsll(a)-1)
#define bitlen(a)  (64-__builtin_clzll(a))

ui32 *inv_mod;
ui64 *isprime_table;

static void usage(void){
    printf("Usage: gap [options]\n");
    printf("\nOptions:\n");
    printf("  -n1 x        First number to check is x\n"); 
    printf("  -n2 y        last number to check is y\n"); 
    printf("  -gap g       searching for gap>=g (default=%d)\n",default_mingap);
    printf("  -delta d     sieving on k*m+[0,d) intervals (default=%d)\n",default_gap_delta);
    printf("  -sb u        sieve uses 2^u bits of memory (default=%d)\n",default_sieve_bits_log2);
    printf("  -bs v        one bucket size is 2^v bytes (default=%d)\n",default_bucket_size_log2);
    printf("  -mem m       the maximal memory usage is m GB (m can be any real number)\n");
    printf("  -t k         use k threads\n");
}

int set_n1;
int set_n2;
int set_gap;
int set_delta;
int set_sb;
int set_bs;
int set_mem;
int set_t;
int set_currentn;
ui64 n0;

ui32 ppi;// number of primes, excluded prime divisors of mod and the small tablesieving primes
ui32 *primes;
ui32 *res_table;

int num_res,r_table[1024];
ui64 *offsets;
int threads;// we will input this

ui64 first_n,last_n;
ui32 primes_per_bucket;
ui32 num_sieve;
ui32 hash3,sh3,hash2,hash0;
ui32 sieve_length_bits_log2;
ui32 num_bucket;
ui32 LEN,LEN64;
ui32 cnt_smallprimes;
ui32 cnt_offsets;
ui32 PROD[64],SIZE[64];
ui32 PPI_131072;
ui32 PPI_LEN;
ui32 *TH;

static const ui64 Bits[64]={// Bits[i]=2^i
  0x0000000000000001,0x0000000000000002,0x0000000000000004,0x0000000000000008,
  0x0000000000000010,0x0000000000000020,0x0000000000000040,0x0000000000000080,
  0x0000000000000100,0x0000000000000200,0x0000000000000400,0x0000000000000800,
  0x0000000000001000,0x0000000000002000,0x0000000000004000,0x0000000000008000,
  0x0000000000010000,0x0000000000020000,0x0000000000040000,0x0000000000080000,
  0x0000000000100000,0x0000000000200000,0x0000000000400000,0x0000000000800000,
  0x0000000001000000,0x0000000002000000,0x0000000004000000,0x0000000008000000,
  0x0000000010000000,0x0000000020000000,0x0000000040000000,0x0000000080000000,
  0x0000000100000000,0x0000000200000000,0x0000000400000000,0x0000000800000000,
  0x0000001000000000,0x0000002000000000,0x0000004000000000,0x0000008000000000,
  0x0000010000000000,0x0000020000000000,0x0000040000000000,0x0000080000000000,
  0x0000100000000000,0x0000200000000000,0x0000400000000000,0x0000800000000000,
  0x0001000000000000,0x0002000000000000,0x0004000000000000,0x0008000000000000,
  0x0010000000000000,0x0020000000000000,0x0040000000000000,0x0080000000000000,
  0x0100000000000000,0x0200000000000000,0x0400000000000000,0x0800000000000000,
  0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000};

static const ui64 InvBits[64]={// InvBits[i]=2^64-1-2^i for masking
0xfffffffffffffffe,0xfffffffffffffffd,0xfffffffffffffffb,0xfffffffffffffff7,
0xffffffffffffffef,0xffffffffffffffdf,0xffffffffffffffbf,0xffffffffffffff7f,
0xfffffffffffffeff,0xfffffffffffffdff,0xfffffffffffffbff,0xfffffffffffff7ff,
0xffffffffffffefff,0xffffffffffffdfff,0xffffffffffffbfff,0xffffffffffff7fff,
0xfffffffffffeffff,0xfffffffffffdffff,0xfffffffffffbffff,0xfffffffffff7ffff,
0xffffffffffefffff,0xffffffffffdfffff,0xffffffffffbfffff,0xffffffffff7fffff,
0xfffffffffeffffff,0xfffffffffdffffff,0xfffffffffbffffff,0xfffffffff7ffffff,
0xffffffffefffffff,0xffffffffdfffffff,0xffffffffbfffffff,0xffffffff7fffffff,
0xfffffffeffffffff,0xfffffffdffffffff,0xfffffffbffffffff,0xfffffff7ffffffff,
0xffffffefffffffff,0xffffffdfffffffff,0xffffffbfffffffff,0xffffff7fffffffff,
0xfffffeffffffffff,0xfffffdffffffffff,0xfffffbffffffffff,0xfffff7ffffffffff,
0xffffefffffffffff,0xffffdfffffffffff,0xffffbfffffffffff,0xffff7fffffffffff,
0xfffeffffffffffff,0xfffdffffffffffff,0xfffbffffffffffff,0xfff7ffffffffffff,
0xffefffffffffffff,0xffdfffffffffffff,0xffbfffffffffffff,0xff7fffffffffffff,
0xfeffffffffffffff,0xfdffffffffffffff,0xfbffffffffffffff,0xf7ffffffffffffff,
0xefffffffffffffff,0xdfffffffffffffff,0xbfffffffffffffff,0x7fffffffffffffff};

static const int add30[8]={6,4,2,4,2,4,6,2};
static const int sub30[8]={2,6,4,2,4,2,4,6};

static const int table_prev_prime[128]={
0,0,2,3,3,5,5,7,7,7,7,11,11,13,13,13,13,17,17,19,19,19,19,23,23,
23,23,23,23,29,29,31,31,31,31,31,31,37,37,37,37,41,41,43,43,43,43,47,47,47,
47,47,47,53,53,53,53,53,53,59,59,61,61,61,61,61,61,67,67,67,67,71,71,73,73,
73,73,73,73,79,79,79,79,83,83,83,83,83,83,89,89,89,89,89,89,89,89,97,97,97,
97,101,101,103,103,103,103,107,107,109,109,109,109,113,113,113,113,113,113,113,113,113,113,113,113,
113,113,127};

static const int table_next_prime[128]={
2,2,2,3,5,5,7,7,11,11,11,11,13,13,17,17,17,17,19,19,23,23,23,23,29,
29,29,29,29,29,31,31,37,37,37,37,37,37,41,41,41,41,43,43,47,47,47,47,53,53,
53,53,53,53,59,59,59,59,59,59,61,61,67,67,67,67,67,67,71,71,71,71,73,73,79,
79,79,79,79,79,83,83,83,83,89,89,89,89,89,89,97,97,97,97,97,97,97,97,101,101,
101,101,103,103,107,107,107,107,109,109,113,113,113,113,127,127,127,127,127,127,127,127,127,127,127,
127,127,127};

// (i*inv64[i])%64==1 for odd i
static const int inv64[64]={0,1,0,43,0,13,0,55,0,57,0,35,0,5,0,47,0,49,0,27,0,61,0,39,0,41,0,19,0,53,
0,31,0,33,0,11,0,45,0,23,0,25,0,3,0,37,0,15,0,17,0,59,0,29,0,7,0,9,0,51,0,21,0,63};

typedef struct{
ui32 offset;
ui32 pr;  // prime number
}bucket;

typedef struct {
ui32 gap;
ui64 p1;
}GAP;
int comp(const void *a,const void *b){
  GAP *gap1=(GAP*)a;
  GAP *gap2=(GAP*)b;
  return(gap2->p1<gap1->p1);
}

void print_time(void){
    time_t timer;
    char w[128];
    struct tm* tm_info;
    time(&timer);
    tm_info=localtime(&timer);
    strftime(w,128,"%Y-%m-%d %H:%M:%S",tm_info);
    printf("%s",w);
}

void print_error_msg(void){printf("Not enough memory!!!\n");exit(1);}
ui64 imax64(ui64 x,ui64 y){if(x>y)return x;return y;}
ui64 imin64(ui64 x,ui64 y){if(x<y)return x;return y;}
ui64 gcd64(ui64 x,ui64 y){if(y==0)return x;return gcd64(y,x%y);}
ui64 lcm64(ui64 x,ui64 y){return (x/gcd64(x,y))*y;}// possibly overflow
ui32 is_power_two(ui64 n){return (n>0&&(n&(n-1))==0);}// return 1 if n is a power of two
ui64 mround_gen(ui64 n,ui64 m){return m*((n+m-1)/m);}
ui64 mround_512(ui64 n){return mround_gen(n,512);}// return the smallest multiple of 512, not smaller than n
ui64 precpower2(ui64 n){return Bits[bitlen(n)-1];}
ui64 nextpower2(ui64 n){if(n<2)return 1;return Bits[bitlen(n-1)];}
ui32 primes2[54]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,
    67,71,73,79,83,89,97,101,103,107,109,113,127,
    131,137,139,149,151,157,163,167,173,179,181,191,
    193,197,199,211,223,227,229,233,239,241,251};// primes up to 256

ui64 conv64(char w[]){
    int i,len=strlen(w);
    for(i=0;i<len;i++)
        if(w[i]=='.'||w[i]=='e'||w[i]=='E'){double d;sscanf(w,"%lf",&d);return ((ui64)d);}
    ui64 v;
    sscanf(w,"%llu",&v);
    return v;
}
    
ui32 lin_solve(ui64 f,ui32 mod,ui32 res,ui32 p){
// Solve (x+f)*mod+res==0 modulo p
// assume that gcd(p,mod)=1
// x==-res/mod-f modulo p
   ui32 r=p%mod;
   ui64 k=(ui64)p*inv_mod[r];
   k=(k-1)/mod;
   r=((ui64)res*k)%p;
   f%=p;
   if(r>=f)r-=f;
   else    r+=p-f;
   return r;
}

static double cpu_time(void){// known small code
  struct rusage r;
  double t;
  getrusage(RUSAGE_SELF,&r);
  t=(double)(r.ru_utime.tv_sec+r.ru_stime.tv_sec);
  t+=1.0e-6*(double)(r.ru_utime.tv_usec+r.ru_stime.tv_usec);
  return t;
}

ui32 isqrt64(ui64 n){// integer square root of n
    
     ui32 ret,v,p2;
     for(ret=0,p2=inf31;p2;p2>>=1){
         v=ret+p2;
         if(((ui64)v*v)<=n)ret=v;
     }
     return ret;
}

ui64 mulmod(ui64 a,ui64 b,ui64 n) {// (a*b)%n
// from mersenneforum.org, with small modification: use mulq,divq instead of imulq,idivq
    ui64 d, dummy; /* d will get a*b mod c */
    asm ("mulq %3\n\t" /* mul a*b -> rdx:rax */
         "divq %4\n\t" /* (a*b)/c -> quot in rax remainder in rdx */
         :"=a"(dummy), "=&d"(d) /* output */
         :"a"(a), "rm"(b), "rm"(n) /* input */
         :"cc" /* imulq and idivq can set conditions */
        );
    return d;
}

ui64 fermatpowmod2_63(ui64 n){// n>0, return (2^(n-1)) mod n, assume that n<2^63
    
    if(n<=64)return (Bits[n-1]%n);
    
    si32 i;
    ui64 n1=n-1;
    ui32 len=bitlen(n1);
    ui64 ret=Bits[n1>>(len-6)]%n,n2=n1>>1;
    
    for(i=len-7;i>=0;i--){
        ret=mulmod(ret,ret,n);
        if((n1>>i)&1){//do ret=(2*ret)%n;
            if(ret>n2)ret=2*ret-n;
            else      ret<<=1;
        }
    }
    return ret;
}

ui64 fermatpowmod2(ui64 n){// return (2^(n-1)) mod n
    
    if(n<inf63)return fermatpowmod2_63(n);
    
    si32 i;
    ui64 n1=n-1;
    ui32 len=bitlen(n1);
    ui64 ret=Bits[n1>>(len-6)]%n,n2=n1>>1;
    
    for(i=len-7;i>=0;i--){
        ret=mulmod(ret,ret,n);
        if((n1>>i)&1){
            if(ret>n2)ret=ret-(n-ret);// to avoid overflow in 64 bits
            else      ret<<=1;
        }
    }
    return ret;
}

ui32 fermatprp2(ui64 n){return (fermatpowmod2(n)==1);}

ui64 next_prime(ui64 n){
    
    if(n<128) return table_next_prime[n];
    if(n>MP64)return 0;// fake, overflow in 64 bits
    
    ui32 i;
    ui64 n2=n+1-(n%30);

    for(i=0;;i++){
        n2+=add30[i&7];
        if(n2>=n&&n2%7&&n2%11&&n2%13&&n2%17&&n2%19&&n2%23&&n2%29&&n2%31&&n2%37&&n2%41&&n2%43&&n2%47&&
          n2%53&&n2%59&&n2%61&&n2%67&&n2%71&&n2%73&&n2%79&&n2%83&&n2%89&&n2%97&&fermatprp2(n2))return n2;
    }
}

ui64 prec_prime(ui64 n){
    
    if(n<128)  return table_prev_prime[n];
    if(n>=MP64)return MP64;
    
    ui32 i;
    ui64 n2=n+31-(n%30);
    for(i=0;;i++){
        n2-=sub30[i&7];
        if(n2<=n&&n2%7&&n2%11&&n2%13&&n2%17&&n2%19&&n2%23&&n2%29&&n2%31&&n2%37&&n2%41&&n2%43&&n2%47&&
          n2%53&&n2%59&&n2%61&&n2%67&&n2%71&&n2%73&&n2%79&&n2%83&&n2%89&&n2%97&&fermatprp2(n2))return n2;
    }
}

si64 single_modinv(si64 a,si64 modulus)
// return by a^(-1) mod modulus, from mersenneforum.org
{/* start of single_modinv */

 a%=modulus;
 if(modulus<0)  modulus=-modulus;
 if(a<0)  a+=modulus;

 si64 ps1, ps2, dividend, divisor, rem, q, t;
 ui32 parity;

 q = 1;
 rem = a;
 dividend = modulus;
 divisor = a;
 ps1 = 1;
 ps2 = 0;
 parity = 0;

 while (divisor > 1)
 {
 rem = dividend - divisor;
 t = rem - divisor;
 if (t >= 0) {
   q += ps1;
   rem = t;
   t -= divisor;
   if (t >= 0) {
     q += ps1;
     rem = t;
     t -= divisor;
     if (t >= 0) {
       q += ps1;
       rem = t;
       t -= divisor;
       if (t >= 0) {
         q += ps1;
         rem = t;
         t -= divisor;
         if (t >= 0) {
           q += ps1;
           rem = t;
           t -= divisor;
           if (t >= 0) {
             q += ps1;
             rem = t;
             t -= divisor;
             if (t >= 0) {
               q += ps1;
               rem = t;
               t -= divisor;
               if (t >= 0) {
                 q += ps1;
                 rem = t;
                 if (rem >= divisor) {
                   q = dividend/divisor;
                   rem = dividend - q * divisor;
                   q *= ps1;
                 }}}}}}}}}
 q += ps2;
 parity = ~parity;
 dividend = divisor;
 divisor = rem;
 ps2 = ps1;
 ps1 = q;
 }

 if(parity==0)
 return (ps1);
 else
 return (modulus - ps1);
}

void init_smallp_segment_sieve(bucket *C,ui64 first_k,ui32 res,ui32 mod){
    
    ui32 i;
    for(i=0;i<ppi;i++){
        C[i].offset=lin_solve(first_k,mod,res,primes[i]);
        C[i].pr=primes[i];}
}

void init_segment_sieve(ui64 first_k,
       ui32 res,ui32 mod,bucket *B,ui32 *previous_bucket,
       ui32 *available_buckets,ui32 *large_lists,ui32 *first_bucket,ui32 *my_head,
       si32 *my_num_available_buckets,ui32 p1,ui32 p2,int thread_id){

   if(p1>p2)return;// no prime
   
   res%=mod;

   ui32 tmp32;
   ui32 i,j,o,o2,p,pos,head=0;
   si32 num_available_buckets=0;
   ui32 size2=num_bucket/primes_per_bucket;

   for(i=0;i<size2;i++)available_buckets[i]=i<<sh3;
   num_available_buckets=size2;
   for(i=0;i<num_sieve;i++)large_lists[i]=0;
   
   ui32 st=((p1>>7)/(8*threads))*8*threads;
   st+=8*thread_id;
   if(st<(p1>>7))st+=8*threads;
   
   int cc=0;
   for(i=0;(ui64)i*LEN<=(ui64)p2;i++)if(TH[i]==thread_id||i<=threads){
       ui32 q1=imax64(p1,i*LEN);
       ui32 q2=imin64(p2,i*LEN+(LEN-1));
       ui32 j1=q1>>7;
       ui32 j2=q2>>7;
       ui32 step,k,mk;
       
       if(i<=threads){j1+=8*thread_id;step=8*threads;mk=8;}
       else          {step=1;mk=1;}
       
       for(j=j1;j<=j2;j+=step)for(k=0;k<mk&&j+k<=(q2>>7);k++){
           
       ui64 temp64=isprime_table[j+k];
       for(p=128*(j+k)+1;temp64;temp64>>=1,p+=2){
           int e=get_lsb(temp64);
           temp64>>=e;
           p+=2*e;
           if(p<q1||p>q2||(mingap2%p==0))continue;
       
           cc++;
           o=lin_solve(first_k,mod,res,p);
       
          o2=o>>sieve_length_bits_log2;
          pos=large_lists[o2];
          if((pos&hash3)==0){
             if(pos==0){
                pos=available_buckets[head];
                large_lists[o2]=pos;
                first_bucket[pos>>sh3]=1;
              
                head++;if(head==size2)head=0;num_available_buckets--;
             }
             else{
                tmp32=pos-primes_per_bucket;
                pos=available_buckets[head];
                large_lists[o2]=pos;
                previous_bucket[pos>>sh3]=tmp32;
                first_bucket[pos>>sh3]=0;
              
                head++;if(head==size2)head=0;num_available_buckets--;
             }
          }
          B[pos].pr=p;
          B[pos].offset=o&hash2;       
          large_lists[o2]++;
   }}}
   assert(num_available_buckets>=0);

   *my_head=head;
   *my_num_available_buckets=num_available_buckets;
   
   return;
}

void sieve_small_primes(ui64 *sieve_array,bucket *C,ui32 num_intervals,
                        ui64 first_k,ui32 res,ui32 mod,ui32 p1,ui32 p2){

    assert(cnt_offsets>0);
    ui32 cnt2,g,h,i,j,k,o,p,st[cnt_offsets];
    ui64 *arr=sieve_array;
    
    if(LEN<131072)cnt2=0;
    else          cnt2=PPI_131072;// PPI_131072 number of primes not used in small sieves up to 2^17

    for(i=0;i<cnt_offsets;i++){
        ui32 P2=PROD[i];
        ui64 inv=single_modinv(mod,P2);
        ui64 pos=((first_k%P2)+mulmod(inv,res%P2,P2))%P2;
        k=((64-(pos&63))*inv64[P2&63])&63;// P2 is odd
        pos+=(ui64)k*P2;
        assert(pos%64==0);
        pos>>=6;
        st[i]=pos;
    }

    for(h=0;h<num_intervals;arr+=LEN64,h++){
        ui64 sh=0;
        for(i=0;i<cnt_offsets;i++){
            ui64 *arr2=arr;
            ui64 *off2=offsets+sh+st[i];
            sh+=SIZE[i]>>6;
            if(i==0){
              for(k=0;k+15<LEN64;){
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
            }
            for(;k<LEN64;k++)arr2[k]=off2[k];}
            else{
            for(k=0;k+15<LEN64;){
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
            }                
            for(;k<LEN64;k++)arr2[k]&=off2[k];}
            st[i]=(st[i]+LEN64)%PROD[i];
        }
        
        ui64 *arr2=arr;
        for(g=0;g<(LEN>>17);arr2+=2048,g++){
            for(j=0;j<PPI_131072;j++){
               o=C[j].offset;
               p=C[j].pr;
               for(k=o;k<131072;k+=p)arr2[k>>6]&=InvBits[k&63];
               C[j].offset=k-131072;
        }}
        for(j=cnt2;j<PPI_LEN;j++){
            o=C[j].offset;
            p=C[j].pr;
            for(k=o;k<LEN;k+=p)arr[k>>6]&=InvBits[k&63];
            C[j].offset=k-LEN;
        }
    }
    
    // correction the offsets for the next block of array
    for(j=0;j<PPI_LEN;j++){
        C[j].offset+=res_table[j];
        if(C[j].offset>=C[j].pr)C[j].offset-=C[j].pr;
    }
    
    return;
}

void segmented_sieve(ui64 *sieve_array,
                     si64 num_intervals,bucket *B,ui32 *previous_bucket,
                     ui32 *available_buckets,ui32 *large_lists,ui32 *first_bucket,ui32 *my_head,
                     si32 *my_num_available_buckets,ui32 *my_start_offset_bucket){
   if(num_intervals<=0)return;
   
   ui32 tmp32;
   ui32 size2=num_bucket/primes_per_bucket;
   ui32 k,o,o1,o2,p,pos,offset_bucket;
   ui32 head=*my_head;
   si32 num_available_buckets=*my_num_available_buckets;
   ui64 *arr=sieve_array;
   ui64 iterations;

   assert(num_available_buckets>=0);
   
   for(iterations=0,offset_bucket=(*my_start_offset_bucket);iterations<num_intervals;arr+=LEN64,
       offset_bucket=(offset_bucket+1)&hash0,iterations++){
       o1=large_lists[offset_bucket];
       
       if(o1!=0){

       ui32 first_pos=((o1-1)>>sh3)<<sh3;// we really need o1-1
       ui32 last_pos=o1;
       ui32 off1=offset_bucket+1;
       large_lists[offset_bucket]=0;
       
       for(;;){
       for(k=first_pos;k<last_pos;k++){
           p=B[k].pr;
           o=B[k].offset;
           
           arr[o>>6]&=InvBits[o&63];
           o+=p-LEN;
           o2=((o>>sieve_length_bits_log2)+off1)&hash0;
           pos=large_lists[o2];
           
          if((pos&hash3)==0){
             if(pos==0){
                pos=available_buckets[head];
                large_lists[o2]=pos;
                // no need to set previous_bucket, first_bucket[] will show
                // that this is the first, there is no previous
                first_bucket[pos>>sh3]=1;
                head++;if(head==size2)head=0;num_available_buckets--;
             }
          else{
              tmp32=pos-primes_per_bucket;
              pos=available_buckets[head];
              large_lists[o2]=pos;
              previous_bucket[pos>>sh3]=tmp32;
              first_bucket[pos>>sh3]=0;
              head++;if(head==size2)head=0;num_available_buckets--;
              }
              if(num_available_buckets<0){
                printf("Bug, out of run in buckets.\n");
                exit(1);
              }
         }
          
           B[pos].pr=p;
           B[pos].offset=o&hash2;
           large_lists[o2]++;
       }
       o1=(o1-1)>>sh3;
       o2=previous_bucket[o1];
       available_buckets[(head+num_available_buckets)%size2]=o1<<sh3;// give back to the list this bucket
       num_available_buckets++;
       
       if(first_bucket[o1])break;// no more bucket for this interval
       first_pos=o2;
       last_pos=o2+primes_per_bucket;
       o1=o2+1;// because there will be (o1-1)>>sh3 we need to do this
       }}
   }
   assert(num_available_buckets>=0);
   
   *my_num_available_buckets=num_available_buckets;
   *my_head=head;
   *my_start_offset_bucket=offset_bucket;
   
   return;
}

ui32 size_isprime;
ui64 size_primes;
void basic_segmented_sieve(ui32 n){// find primes up to max(n,65536)
    
    ui32 LEN2=imax64(LEN,65536);
    n=imax64(n,LEN2);
    n-=n%65536;
    n+=65535;

    ui32 i,j,k,m,m2,p,plist[6542],offset[6542];
    ui64 temp64,a[512];

    // First get the primes up to 65536
    for(i=0;i<512;i++)a[i]=inf64;
    a[0]&=InvBits[0];
    for(i=1;i<128;i++)if(a[i>>6]&Bits[i&63]){
        p=2*i+1;
        for(j=(p*p-1)>>1;j<32768;j+=p)a[j>>6]&=InvBits[j&63];
    }
    
    plist[0]=2;ppi=1;
    for(i=0;i<32768;i++)if(a[i>>6]&Bits[i&63])plist[ppi++]=2*i+1;
    printf("primepi(65536)=%d;\n",ppi);
    assert(ppi==6542);
    
    size_isprime=mround_512(8*((n>>7)+1));
    if(posix_memalign((void**)&isprime_table,ALIGNEMENT,size_isprime)!=0)print_error_msg();
    
    for(i=0;i<=(n>>7);i++)isprime_table[i]=inf64;
    isprime_table[0]&=InvBits[0];
    for(i=1;i<ppi;i++)offset[i]=(plist[i]*plist[i]-1)>>1;
    for(i=0;i<=(n>>16);i++){
        m=32768*i+32767;
        m2=2*m+1;
        for(j=1;j<6542;j++){
            p=plist[j];
            if(p*p>m2)break;
            for(k=offset[j];k<=m;k+=p)
                isprime_table[k>>6]&=InvBits[k&63];
            offset[j]=k;
        }
    }
    
    ui32 cnt=0;
    for(i=0;i<=(LEN2>>7);i++)
       {for(temp64=isprime_table[i];temp64;temp64>>=1)cnt+=temp64&1;}
    cnt=mround_512(cnt);
    ui64 size_primes=(ui64)cnt*sizeof(ui32);
    if(posix_memalign((void**)&primes,ALIGNEMENT,size_primes)!=0)print_error_msg();
    ppi=0;
    for(i=0;i<=(LEN2>>7);i++){
       for(p=128*i+1,temp64=isprime_table[i];temp64;p+=2,temp64>>=1)if(temp64&1){
           if(p<128||p>LEN2||mingap2%p==0)continue;
           primes[ppi++]=p;
       }
    }
    printf("done basic segment sieve\n");
    
    return;
}

void get_params(void){
    
    int ret;
    ui64 v;
    char u[256],w[256];
    
    FILE *fin;
    fin=fopen("worktodo_gap.txt","r");
    
    if(fin!=NULL){
        printf("Found unfinished work, do you want to continue that (y/n) ? ");
        ret=scanf("%s",w);
        if(w[0]=='y'||w[0]=='Y'){
            ret=fscanf(fin,"%s",w);sscanf(w,"n1=%s",u);v=conv64(u);
                if(set_n1&&v!=first_n){
                printf("You given a different n1 with the -n1 switch from what is in the worktodo file,\nthat would mean a new range, not a continuation of the previous work. We exit.\n");
                exit(1);}first_n=v;set_n1=1;
            
            ret=fscanf(fin,"%s",w);sscanf(w,"n2=%s",u);v=conv64(u);
                if(set_n2&&v!=last_n){
                printf("You given a different n2 with the -n2 switch from what is in the worktodo file,\nthat would mean a new range, not a continuation of the previous work. We exit.\n");
                exit(1);}last_n=v;set_n2=1;
            
            ret=fscanf(fin,"%s",w);if(!set_currentn){sscanf(w,"n=%llu",&n0);set_currentn=1;}//there is no switch to set currentn
            ret=fscanf(fin,"%s",w);if(!set_gap){sscanf(w,"gap=%d",&mingap);set_gap=1;}
            ret=fscanf(fin,"%s",w);if(!set_delta){sscanf(w,"delta=%d",&gap_delta);set_delta=1;}
            ret=fscanf(fin,"%s",w);if(!set_sb){sscanf(w,"sb=%d",&sieve_bits_log2);set_sb=1;}
            ret=fscanf(fin,"%s",w);if(!set_bs){sscanf(w,"bs=%d",&bucket_size_log2);set_bs=1;}
        }
        else remove("worktodo_gap.txt");
        fclose(fin);
    }
    
    if(!set_n1){
        printf("Give the first tested number! ");
        ret=scanf("%s",w);first_n=conv64(w);
        set_n1=1;
    }
    if(first_n<=1+inf32){printf("Too small, n1>2^32 should be true.\n");exit(1);}

    if(!set_n2){
        printf("Give the last tested number!  ");
        ret=scanf("%s",w);last_n=conv64(w);
        set_n2=1;
    }
    if(last_n>=inf64-inf32){printf("Too large, n2<2^64-2^32 should be true.\n");exit(1);}

    if(!set_gap){
        printf("Give the minimal gap value! (default=%d) ",default_mingap);
        ret=scanf("%d",&mingap);
        set_gap=1;
    }
    
    if(!set_delta){
        printf("Give the delta value for gap! (default=%d) ",default_gap_delta);
        ret=scanf("%d",&gap_delta);
        set_delta=1;
    }
    
    if(!set_sb){
        printf("Give the log2 of the total sum of sieving bits (default=%d) ",default_sieve_bits_log2);
        ret=scanf("%d",&sieve_bits_log2);
        set_sb=1;
    }
    
    if(!set_bs){
        printf("Give the log2 size of one bucket's byte size (default=%d) ",default_bucket_size_log2);
        ret=scanf("%d",&bucket_size_log2);
        set_bs=1;
    }
    
    if(sieve_bits_log2<16||bucket_size_log2>16||sieve_bits_log2<bucket_size_log2){// smart check
        printf("Do you really want this(?), not swapped sb-bs values?\n");
        printf("sb is log2 of the sieving size in bits.\n");
        printf("bs is log2 of the one bucket size in bytes.\n");
        printf("You can still use Ctrl+C to break from here.\n");
    }
    
    if(!set_mem){
       printf("Give for this code the available memory in GB (1GB=2^30 bytes),\nthis can be a real number (say 4.35) ");
       ret=scanf("%lf",&max_memory_gigabytes);
       set_mem=1;
    }
    
    int maxthreads=omp_get_max_threads();
    if(!set_t){
       printf("Found %d thread(s) on this computer,\n",maxthreads);
       printf("give the number of availables thread(s) for this code! ");
       ret=scanf("%d",&threads);
       set_t=1;
    }
    assert(threads>=0&&threads<=maxthreads);
    if(threads==0)exit(1);
    omp_set_num_threads(threads);
    
    assert(first_n<=last_n);
    
    return;
}

ui64 size_offset;
int inits(void){// return 1 if all inits is success
    
    get_params();
    
    ui32 prime_per_thread[threads];
    
    ui32 maxp=imax64(isqrt64(last_n),131072);
    sieve_length_bits_log2=bitlen(Bits[sieve_bits_log2]/threads)-1;
    while(sieve_length_bits_log2>6&&sieve_length_bits_log2>=bitlen(maxp))sieve_length_bits_log2--;
    LEN=Bits[sieve_length_bits_log2];

    assert(bitlen(maxp)>=sieve_length_bits_log2+1);// with this maxp>=LEN is true
    num_sieve=Bits[bitlen(maxp)-sieve_length_bits_log2];
    hash0=num_sieve-1;
    
    primes_per_bucket=precpower2((1<<bucket_size_log2)/(2*size_ui32));
    hash3=primes_per_bucket-1;
    sh3=bitlen(hash3);
    hash2=Bits[sieve_length_bits_log2]-1;

    LEN64=LEN>>6;
    hash2=LEN-1;

    ui32 h,i,j,k,p,temp32,cnt;
    ui64 I;
    double eff,best_eff;
    time_t sec=time(NULL);
    
    assert(sizeof(ui32)>=4);
    assert(sizeof(ui64)==8);
    
    temp32=30;
    mingap2=mingap-gap_delta;
    best_eff=1.0;    
    for(i=30;i<=mingap2;i+=30){
        eff=(double)1/i;
        for(j=0;j<31;j++)
            if(i%primes2[j]==0)eff*=(double)(primes2[j]-1)/primes2[j];
        if(i==30||eff<best_eff){
           temp32=i;
           best_eff=eff;
        }
    }
    mingap2=temp32;
    printf("Using mingap2=%d;\n",mingap2);
    
    basic_segmented_sieve(imax64(maxp,LEN));
    
    ui32 mod=mingap2;
    ui32 size_mod=mround_512(mod)*sizeof(ui32);
    if(posix_memalign((void**)&inv_mod,ALIGNEMENT,size_mod)!=0)print_error_msg();
    for(i=0;i<mod;i++)inv_mod[i]=single_modinv(i,mod);
   
    num_res=0;
    for(i=0;i<gap_delta;i++)
        if(gcd64(i,mingap2)==1)r_table[num_res++]=i;
    printf("num_res=%d;\n",num_res);

    ui64 size,len=0;
    cnt=0;
    for(i=0;i<31;){
        if(mingap2%primes2[i]>0){
          ui32 prod=primes2[i];
          i++;
          for(;i<31&&prod<MAX_SIZE/8/primes2[i];){
             if(mingap2%primes2[i]>0)prod*=primes2[i];
             i++;}
             
          size=mround_512(lcm64(prod,64)+LEN);
          len+=size;
          PROD[cnt]=prod;
          SIZE[cnt]=size;
          cnt++;
       }
       else i++;
    }
    size_offset=mround_512(len/8);
    if(posix_memalign((void**)&offsets,ALIGNEMENT,size_offset/8*sizeof(ui64))!=0)print_error_msg();

    for(I=0;I<(len>>6);I++)offsets[I]=inf64;// Here assume that sizeof(ui64)=8
    ui64 sh=0;
    for(i=0;i<cnt;i++){
        ui64 *offsets3=offsets+sh;
        size=SIZE[i];
        for(j=0;j<31;j++)if(PROD[i]%primes2[j]==0){
            p=primes2[j];
            for(I=0;I<size;I+=p)offsets3[I>>6]&=InvBits[I&63];
        }
        sh+=size/64;
    }
    cnt_offsets=cnt;
    
    cnt=0;
    for(i=0;i<threads;i++)prime_per_thread[i]=0;
    PPI_131072=0;
    PPI_LEN=0;
    for(i=0;i<=(maxp>>7);i++){
        ui64 temp64=isprime_table[i];
        for(p=128*i+1;temp64;temp64>>=1,p+=2){
            int e=get_lsb(temp64);
            temp64>>=e;
            p+=2*e;
            if(p>128&&mingap2%p>0){
               if(p<131072)PPI_131072++;
               if(p<LEN)  PPI_LEN++;
            }
        }
    }
    
    // Distribute the primes in [LEN,maxp] in such a way that in each thread sum(1/p) will be roughly equal
    //    to get roughly equal time in (the bucket version) segmented sieve
    //    and for a multi threaded run the larger primes will be in blocks in the threads:
    //        giving that (o+p)/LEN will evaluate as a very few different values
    //    it looks like that it will balance also the number of primes in each thread
    TH=(ui32*)malloc((maxp/LEN+1)*sizeof(ui32));
    double pr[threads];
    for(i=0;i<threads;i++)pr[i]=0.0;
    for(i=1;(ui64)i*LEN<=(ui64)maxp;i++){
        if(i<=threads){
            ui32 p1=LEN*i;
            ui32 p2=p1+(LEN-1);
            ui32 j1=p1>>7;
            ui32 j2=p2>>7;
            for(k=0;k<threads;k++)for(j=j1+8*k;j<=j2&&j<=(maxp>>7);j+=8*threads)for(h=0;h<8&&h+j<=(maxp>>7);h++){
                ui64 tmp64=isprime_table[h+j];
                p=128*(h+j)+1;
                for(;tmp64;tmp64>>=1,p+=2){
                    int e=get_lsb(tmp64);
                    tmp64>>=e;
                    p+=2*e;
                    if(p<128||p>maxp||mingap2%p==0)continue;
                    pr[k]+=(double)1/p;
                    prime_per_thread[k]++;
                }
            }
        }
        else{        
            ui32 p1=LEN*i,pos=0;
            double low_pr;
            double s=0.0;
            cnt=0;
            for(j=1;j<LEN;j+=2){
               p=p1+j;
               pos=(p-1)>>1;
               if(p<128||p>maxp||mingap2%p==0||(isprime_table[pos>>6]&Bits[pos&63])==0)continue;
               cnt++;
               s+=(double)1/p;
            }
            for(j=0;j<threads;j++)
                if(j==0||pr[j]<low_pr){pos=j;low_pr=pr[j];}
            pr[pos]+=s;
            TH[i]=pos;
            prime_per_thread[pos]+=cnt;
        }
    }
    ui32 np=0;
    for(i=0;i<threads;i++)np+=prime_per_thread[i];
    
    num_bucket=1;
    for(i=0;i<threads;i++)num_bucket=imax64(num_bucket,prime_per_thread[i]);
    num_bucket+=(num_sieve+3)*primes_per_bucket;// it is enough +1 for rounding and +1 for the current bucket
                                                // +1 for a new bucket
    num_bucket=mround_gen(num_bucket,lcm64(512,primes_per_bucket));
    printf("All initializations are done in %ld seconds.\n",time(NULL)-sec);
    
    return 1;
}

void ff(void){
    
    time_t sec=time(NULL);

    ui32 res,maxp=isqrt64(last_n);

    assert(is_power_two(num_sieve)==1);
    assert(sieve_length_bits_log2>=6&&sieve_length_bits_log2<32);

    ui32 np=num_bucket/primes_per_bucket;
    ui64 size=(ui64)threads*num_bucket;
    ui64 size2=(ui64)threads*num_sieve;
    ui64 size3=(ui64)threads*((ui64)count_LEN_intervals*(2*LEN/8));
    ui64 size6=(ui64)threads*np;
    ui64 size8=mround_512(ppi);
    ui64 size9=(ui64)threads*mround_512(ppi);
    
    double m0=0.0,m1=0.0;
    m0+=(double)size_offset;// size of offsets array
    m0+=(double)size_isprime;// size of the isprime array
    m0+=(double)size_primes;// size of the primes array
    m0+=(double)size8*sizeof(ui32);// size of the res_table array
    
    m1+=(double)num_bucket*sizeof(bucket);// B
    m1+=(double)np*sizeof(ui32);// available_buckets
    m1+=(double)num_sieve*sizeof(ui32);// large_lists
    m1+=(double)count_LEN_intervals*(2*LEN/8);// sieve_array
    m1+=(double)np*sizeof(ui32);// previous bucket
    m1+=(double)np*sizeof(ui32);// first bucket
    m1+=(double)mround_512(ppi)*sizeof(bucket);// C

    m1*=(double)threads;
    
    double mbytes=(double)DP30*max_memory_gigabytes;
    mbytes-=m0+m1;
    if(save_nextprimetest)mbytes/=(double)LEN/4;
    else                  mbytes/=(double)LEN/8;
    mbytes-=0.5;
    si64 si_temp=(si64)mbytes;
    ui64 ilow=imax64(count_LEN_intervals,(ui64)nextpower2(maxp)/LEN);
    
    if(si_temp<1||si_temp<ilow){
       m0+=(double)ilow*(LEN/8);
       if(save_nextprimetest)m0+=(double)ilow*(LEN/8);
       m0/=DP30;m0+=0.005;
       m1/=DP30;m1+=0.005;
       printf("You have given too few memory, need at least %.2lf GB of memory. \n",m0+m1);
       exit(1);
    }
    ui64 num_iterations=mround_gen((ui64)si_temp,count_LEN_intervals);
    
    ui64 size4=(ui64)num_iterations*(LEN/8);// size of ans, saved ans
    m0+=(double)size4;
    if(save_nextprimetest)m0+=(double)size4;
    m0/=DP30;
    m1/=DP30;
    used_memory=m0+m1;
    printf("Memory usage: %.2lf GB.\n",used_memory);
    
    ui64 *ans,*saved_ans,*sieve_array;
    ui32 *large_lists,*previous_bucket,*available_buckets,*first_bucket;
    bucket *B,*C;
    if(posix_memalign((void**)&B,ALIGNEMENT,size*sizeof(bucket))!=0)print_error_msg();
    if(posix_memalign((void**)&C,ALIGNEMENT,size9*sizeof(bucket))!=0)print_error_msg();
    
    if(posix_memalign((void**)&available_buckets,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&large_lists,ALIGNEMENT,size2*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&sieve_array,ALIGNEMENT,size3)!=0)print_error_msg();
    if(posix_memalign((void**)&ans,ALIGNEMENT,size4)!=0)print_error_msg();
    if(save_nextprimetest){if(posix_memalign((void**)&saved_ans,ALIGNEMENT,size4)!=0)print_error_msg();}
    else saved_ans=(ui64*)malloc(1*sizeof(ui64));
    if(posix_memalign((void**)&previous_bucket,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&first_bucket,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&res_table,ALIGNEMENT,size8*sizeof(ui32))!=0)print_error_msg();

    ui64 first_k;
    if(!set_currentn){n0=first_n;set_currentn=1;
        first_k=mround_gen(n0,mingap2)/mingap2-1;}
    else first_k=n0/mingap2;
    ui64 last_k=last_n/mingap2;
    
    int mod=mingap2,i,j;
    
    ui64 I;
    ui64 large_block=(ui64)count_LEN_intervals*LEN;
    ui64 processed_large_blocks=0;
    
    int first_run=1;
    ui32 mid_res=0;
    
    printf("Program version number=%s;\n",version);
    printf("Start the main algorithm; date: ");print_time();
    printf("\ninterval=[%llu,%llu]; now at n=%llu;\n",first_n,last_n,n0);
    printf("gap=%d;delta=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB\n",
           mingap,gap_delta,sieve_bits_log2,bucket_size_log2,threads,max_memory_gigabytes);    
    
    ui64 step_k=(ui64)num_iterations*LEN;// num_iterations is divisible by count_LEN_intervals
    ui32 ppi2=mround_512(ppi);
    
    ui64 temp=(ui64)large_block*(threads-1);
    for(i=0;i<ppi;i++){
        res_table[i]=primes[i]-(temp%primes[i]);
        if(res_table[i]==primes[i])res_table[i]=0;
    }
    
    for(;first_k<=last_k;first_k+=step_k){
        if(!first_run){
        FILE* fout;
        fout=fopen("worktodo_gap.txt","w");
        fprintf(fout,"n1=%llu\n",first_n);
        fprintf(fout,"n2=%llu\n",last_n);
        fprintf(fout,"n=%llu\n",(ui64)first_k*mod);
        fprintf(fout,"gap=%d\n",mingap);
        fprintf(fout,"delta=%d\n",gap_delta);
        fprintf(fout,"sb=%d\n",sieve_bits_log2);
        fprintf(fout,"bs=%d\n",bucket_size_log2);
        fclose(fout);}
        first_run=0;
        
        ui64 num_large_block=imin64(num_iterations/count_LEN_intervals,
                mround_gen(last_k-first_k+1,large_block)/large_block);
        ui64 I2=((ui64)num_large_block*large_block)/64;
        
        for(I=0;I<I2;I++)ans[I]=0;
        
        temp=imin64(first_k+step_k,last_k);
        temp=imin64((temp+1)*mod,last_n);
        maxp=isqrt64(temp);
        
        int saved=0;// note: it is possible that there will be no save (if num_res<2)
        
        for(i=0;i<num_res;i++){
            if(save_nextprimetest&&i==(num_res+1)/2){
                ui64 I2=((ui64)num_iterations*LEN)/64;
                for(I=0;I<I2;I++)saved_ans[I]=ans[I];
                saved=1;
                mid_res=r_table[i-1];
            }
            
            res=r_table[i];
            ui32 head[threads];
            ui32 start_offset_bucket[threads];
            si32 num_available_buckets[threads];
            
            ui64 it;
            for(it=0;it<num_large_block+threads-1;it++){
                #pragma omp parallel for schedule(dynamic,1)
                for(j=0;j<threads;j++){

                ui32 id2=(it+2*threads-j)%(2*threads); 
                ui32 id=(id2+threads)%(2*threads);
                
                if(id==threads)id=0;
                else if(id==0)id=threads;
                
                bucket *B2=B+(ui64)j*num_bucket;
                bucket *C2=C+(ui64)(id2%threads)*ppi2;
                ui32 *previous_bucket2=previous_bucket+(ui64)j*np;
                ui32 *available_buckets2=available_buckets+(ui64)j*np;
                ui32 *large_lists2=large_lists+(ui64)j*num_sieve;
                ui32 *first_bucket2=first_bucket+(ui64)j*np;
                ui64 *sieve_array2=sieve_array+(ui64)id2*(large_block/64);
                ui64 *sieve_array3=sieve_array+(ui64)id*(large_block/64);
                
                ui64 first_k2=first_k+(ui64)(it+(id%threads))*large_block;

                if(it==0){// init the sieves
                    init_smallp_segment_sieve(C2,first_k2,res,mod);
                    
                    start_offset_bucket[j]=0;
                    head[j]=0;
                    num_available_buckets[j]=0;
                    init_segment_sieve(first_k,res,mod,B2,previous_bucket2,
                       available_buckets2,large_lists2,first_bucket2,&head[j],
                       &num_available_buckets[j],LEN,maxp,j);
                }
                
                if(it%threads==0&&it<num_large_block){// do the smallprime sieve
                   sieve_small_primes(sieve_array3,C2,count_LEN_intervals,
                   first_k2,res,mod,128,LEN);
                }
                
                // do the (it-j)-th block
                if(it>=j&&it-j<num_large_block){
                    segmented_sieve(sieve_array2,
                       count_LEN_intervals,B2,previous_bucket2,
                       available_buckets2,large_lists2,first_bucket2,&head[j],
                       &num_available_buckets[j],&start_offset_bucket[j]);
                    
                    if(j==threads-1){//save the large block, note that only one thread
                       // worked on this block, so there is no race
                       //printf("start or\n");
                       ui64 K,K2=large_block/64;
                       ui64 *ans2=ans+(ui64)K2*(it-j);// j=threads-1
                       for(K=0;K<K2;K++)ans2[K]|=sieve_array2[K];
                }}}}
        }
        
        ui64 cnt_findprp_prime[threads],cnt_empty_block[threads];
        
        ui64 *sols,K;
        int nt=omp_get_max_threads();// max. number of threads
        ui32 num_solutions[nt];
        sols=(ui64*)malloc(2*nt*MAX_NUM_SOLUTIONS*sizeof(ui64));
        for(i=0;i<nt;i++){
            num_solutions[i]=0;
            cnt_findprp_prime[i]=0;
            cnt_empty_block[i]=0;}
        
        #pragma omp parallel for schedule(dynamic,1)
        for(K=0;K<I2;K+=1048576){
            int th_id=omp_get_thread_num();
            ui32 h;
            ui64 G,k2=imin64(K+1048576,I2);
            ui64 range_k=k2-K;
            ui64 *ans2=ans+K;
               for(G=0;G<range_k;G++)if(ans2[G]!=inf64){
                   ui64 tmp64=~ans2[G];
                   for(h=0;tmp64;tmp64>>=1,h++){// so now hunting for bit=1
                       if((tmp64&0xffffffff)==0){tmp64>>=32;h+=32;}
                       if((tmp64&0xffff)==0)    {tmp64>>=16;h+=16;}
                       if((tmp64&0xff)==0)      {tmp64>>=8;h+=8;}
                       if((tmp64&0xf)==0)       {tmp64>>=4;h+=4;}
                       if((tmp64&0x3)==0)       {tmp64>>=2;h+=2;}
                       if((tmp64&0x1)==0)       {tmp64>>=1;h++;}  
                       ui64 pos=64*(K+G)+h;
                       
                       ui64 mult=first_k+pos;
                       if(mult>=first_k&&mult<=last_k){
                          ui64 n=(ui64)mult*mod;// n is even
                          // there is no prime in [n,n+gap_delta)
                          ui64 p1=prec_prime(n),p2;
                          cnt_findprp_prime[th_id]++;
                          cnt_empty_block[th_id]++;
                          if(saved&&pos+1<64*I2&&(saved_ans[(pos+1)>>6]&Bits[(pos+1)&63])){
                              p2=n+mod+mid_res;
                              if(p2-p1<mingap)continue;
                          }
                          p2=next_prime(n+gap_delta);
                          cnt_findprp_prime[th_id]++;
                          if(p2-p1>=mingap||p2-p1>=default_report_gap){
                             ui32 o=2*th_id*MAX_NUM_SOLUTIONS+2*num_solutions[th_id];
                             sols[o]=p2-p1;
                             sols[o+1]=p1;
                             num_solutions[th_id]++;
                             if(num_solutions[th_id]==MAX_NUM_SOLUTIONS){
                                 ui32 k;
                                 o=2*th_id*MAX_NUM_SOLUTIONS;
                                 FILE* fout;
                                 fout=fopen("gap_solutions.txt","a+");
                                 for(k=0;k<num_solutions[th_id];k++){
                                     printf("%llu %llu\n",sols[o+2*k],sols[o+2*k+1]);
                                     fprintf(fout,"%llu %llu\n",sols[o+2*k],sols[o+2*k+1]);
                                 }
                                 fclose(fout);
                                 num_solutions[th_id]=0;
                             }                              
                          }
                       }
                   }
               }
            }
        ui64 c0=0,c1=0;
        for(i=0;i<nt;i++)
        {c0+=cnt_findprp_prime[i];c1+=cnt_empty_block[i];}
        
        ui32 ns=0;for(i=0;i<nt;i++)ns+=num_solutions[i];
        GAP my_gap[ns];
        
        ns=0;
        for(j=0;j<nt;j++){
            ui32 k,o=2*j*MAX_NUM_SOLUTIONS;
            for(k=0;k<num_solutions[j];k++){
                my_gap[ns].gap=(ui32)sols[o+2*k];
                my_gap[ns].p1=sols[o+2*k+1];
                ns++;
            }
        }
        
        //qsort(my_gap,ns,sizeof(GAP),comp);
        int l1,l2;
        // bubble sort
        for(l1=0;l1<ns;l1++)for(l2=0;l1+l2+1<ns;l2++)if(my_gap[l2].p1>my_gap[l2+1].p1){
            ui32 t1=my_gap[l2].gap;my_gap[l2].gap=my_gap[l2+1].gap;my_gap[l2+1].gap=t1;
            ui64 t2=my_gap[l2].p1; my_gap[l2].p1=my_gap[l2+1].p1;  my_gap[l2+1].p1=t2;}
        
        FILE* fout;
        fout=fopen("gap_solutions.txt","a+");
        for(i=0;i<ns;i++)if(i==0||my_gap[i].p1>my_gap[i-1].p1){
            printf("%u %llu\n",my_gap[i].gap,my_gap[i].p1);
            fprintf(fout,"%u %llu\n",my_gap[i].gap,my_gap[i].p1);
        }
        fclose(fout);
        free(sols);
        
        processed_large_blocks+=num_large_block;
        double rate=(double)processed_large_blocks/((double)(time(NULL)-sec)+0.01);// to avoid division by 0
        rate*=(double)large_block;
        rate*=(double)mod;
        
        ui64 nn=first_k+(ui64)num_large_block*large_block;
        if(nn>last_k)nn=last_k+1;
        nn*=mod;
        nn=imin64(nn,last_n);
        
        printf("%.2lfe9 n/sec.;now we are at n=%llu;time=%ld sec.; date:  ",rate/1e9,nn,time(NULL)-sec);
        print_time();printf("\n");
        
        fout=fopen("gap_log.txt","a+");
        fprintf(fout,"Done interval=[%llu,%llu] with version=%s;gap=%d;delta=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            imax64(first_n,(ui64)first_k*mod),nn,version,mingap,gap_delta,sieve_bits_log2,bucket_size_log2,threads,max_memory_gigabytes);    
        fclose(fout);
    }
    FILE* fout;
    fout=fopen("results_gap.txt","a+");
    fprintf(fout,"Done interval=[%llu,%llu] with version=%s;gap=%d;delta=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            first_n,last_n,version,mingap,gap_delta,sieve_bits_log2,bucket_size_log2,threads,max_memory_gigabytes);    
    fclose(fout);
    remove("worktodo_gap.txt");

    free(ans);
    free(saved_ans);
    free(sieve_array);
    free(B);
    free(available_buckets);
    free(previous_bucket);
    free(large_lists);
    free(first_bucket);
    free(offsets);
    free(inv_mod);
    free(primes);
    free(isprime_table);
    free(TH);
    free(C);
    free(res_table);
    
    return;
}


int main(int argc, char **argv){

    set_n1=0;
    set_n2=0;
    set_gap=0;
    set_delta=0;
    set_sb=0;
    set_bs=0;
    set_mem=0;
    set_t=0;
    set_currentn=0;
    
    while((argc>1)&&(argv[1][0]=='-')){
      if(strcmp(argv[1],"-h")==0||strcmp(argv[1],"--help")==0){
        usage();
        exit(1);
      }
      else if(argc>2&&strcmp(argv[1],"-n1")==0){
	    first_n=conv64(argv[2]);
	    set_n1=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-n2")==0){
	    last_n=conv64(argv[2]);
	    set_n2=1;
	    argv+=2;
	    argc-=2;
	  }
	  else if(argc>2&&strcmp(argv[1],"-gap")==0){
        mingap=atoi(argv[2]);
        set_gap=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-delta")==0){
        gap_delta=atoi(argv[2]);
        set_delta=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-sb")==0){
        sieve_bits_log2=atoi(argv[2]);
        set_sb=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-bs")==0){
        bucket_size_log2=atoi(argv[2]);
        set_bs=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&(strcmp(argv[1],"-mem")==0||strcmp(argv[1],"-memory")==0)){// also accept the -memory switch
        max_memory_gigabytes=atof(argv[2]);// convert to double
        set_mem=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-t")==0){
        threads=atoi(argv[2]);
        set_t=1;
        argv+=2;
        argc-=2;
      }
      else{
	    fprintf(stderr,"Unknown option: %s\n",argv[1]);
	    exit(1);
	  }
    }
    
    inits();
    time_t sec=time(NULL);
    double dt=cpu_time();
    ff();
    printf("Done interval=[%llu,%llu]\nwith gap=%d;delta=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            first_n,last_n,mingap,gap_delta,sieve_bits_log2,bucket_size_log2,threads,max_memory_gigabytes);    
    printf("Search used %ld sec. (Wall clock time), %.2lf cpu sec.\n",time(NULL)-sec,cpu_time()-dt);
    print_time();printf("\n");
    
    return 0;
}
