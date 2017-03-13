#include <stdio.h>
#include <time.h>
__global__ void ken(double *a)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  a[id]=pow((double)(4*id+1),-1)-pow((double)(4*id+3),-1);
}
__global__ void ken2(double *a,double *b,int *dcount)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  int count=*dcount;
  if(count%2==0)
  {
    count=count/2;
  }
  else
  {
    count=count/2+1;
  }
  if(id<count)
  {
    b[id]=a[id*2]+a[id*2+1];
    a[id*2]=0.0;
    a[id*2+1]=0.0;  
  }
  if(id==0)
    *dcount=count;
}
int main()
{
  clock_t t1, t2;
  int block =50000;
  int thread=300;
  int count=block*thread;
  int size=block*thread;
  int *dcount;
  double *a,*da,*db;
  a=(double*)malloc(size*sizeof(double));
  cudaMalloc((void**)&da,size*sizeof(double));
  cudaMalloc((void**)&db,size*sizeof(double));
  cudaMalloc((void**)&dcount,sizeof(int));
 
  t1 = clock();
  cudaMemcpy(dcount,&count,sizeof(int),cudaMemcpyHostToDevice);
  ken<<<block,thread>>>(da);
  while(count>1)
  {
    if(count%2==0)
      count=count/2;
    else
      count=count/2+1;
    ken2<<<block,thread>>>(da,db,dcount);
    cudaDeviceSynchronize();
    cudaMemcpy(da,db,size*sizeof(double),cudaMemcpyDeviceToDevice);

  }
  cudaMemcpy(a,da,size*sizeof(double),cudaMemcpyDeviceToHost);
  t2 = clock();
  printf("\na[0]=%.8lf\n size=%d\n",4*a[0],size*2);
  printf("%lf\n", (t2-t1)/(double)(CLOCKS_PER_SEC));
  return 0;
}
