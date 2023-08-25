//~ - Final MST Added ! - bug fixed
/*  - 2N Threads of two sssps EVEN+ODD - done
 *  - DONE: KN THREADS
 *  - ATTEMPT: KN THREADS SH MEM
 *  - ATTEMPT: CLEANUP
 *  - CudaStream k=2 SSSP
 *  - Code Clean - done in v5
 *  - Explore opportunities for CPU computation before/after kernel launches
 *  - Preprocessing ?
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "CUDAMST.cu"
#include <set>
#include <map>
#include <vector>
#include <unordered_map>
using namespace std;

#define MAX_INT_IN_SHARED_PER_BLOCK 12288
#define SH_REGS_PER_THREAD 24

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(1); \
 }                                                                 \
}
#define TRUE 1
#define FALSE 0

//~ #define MAX_THREADS_PER_BLOCK 1024 // It is defined in CUDAMST.h

#define MAX_COST 1073741823
#define LEVEL 0
// 0 submit level-- no print
// 1 debug level -- prints as needed.

#define DEBUG if(LEVEL)

cudaEvent_t tstart, tstop;
float totalTimeMilliSec = 0.0;
int sCount = 2; //DEFAULT

/*
__global__ void cpyParentArray(int N, int index, int *d_parentArrays, int *d_parent ){
	unsigned id =  blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(id < N){
		d_parentArrays[ N*index + id] = d_parent[id];
		//~ printf("G:P[%d]=dP[%d]=%d\n", id+index,id,d_parent[id]);
	}
}
*/

__global__ void cpyParentArrayNew(int N, int index, int *d_parentArrays, int *d_parent, int sCount, int tempScount){
	unsigned id =  blockIdx.x*blockDim.x + threadIdx.x; // Changed
	if(id < tempScount*N){
      d_parentArrays[ sCount*N*index + id] = d_parent[id]; //next block after sCount*N many
	}
}

__global__ void kernelInitDistAndParent(int N, int *minDist, int* parent, int tempScount){ //FOR KSSSP

	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < tempScount*N) {
		minDist[id]=INT_MAX/2;
		parent[id] = -1;
	}
}

__global__ void kernelInitSources(int N, int* source, int *minDist, int tempScount){ //FOR KSSSP
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id == 0){
    for (int ii=0; ii < tempScount; ++ii)
      minDist[ N*ii + source[ii] ] = 0 ;
  }
}


__global__ void csrKernelBellmanFordMoore(int N,  int* source, // K SSSP  PULLL//PULLL
	int* csrM, int* csrD, int* csrW,
	bool* changed,
	int *minDist, int *parent,
  int sCount,
  int tempScount
  ){

	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id < tempScount*N){



		int u = id;   // may not be needed but easier to code! -- ok! .For reading output
		int uIn = id % N;   // for read from input
		int start = csrM[uIn];  // may not be needed but easier to code! --refereed once :P
		int end= csrM[uIn+1];
		int i, v, old, newDist, minSize;

    // variables to implement ShMEM
		int size = end - start;		//adjList size per thread
    int j;
    __shared__ int shD[MAX_INT_IN_SHARED_PER_BLOCK]; // for now. 12288  min(12288, _2M) -- it can never spill over


    /***
    for(i = start; i< end; ++i) { // u and v is adj[u] // Note each nbr is sequential
      v = csrD[i];	//Read

      /// relax
      newDist =  minDist[(id/N)*N+v]+ csrW[i];      // Generalized version KN

      //~ if ((u < N)) { //2 SSSP
        //~ newDist =  minDist[v]+ csrW[i];
      //~ }else{
       //~ newDist =  minDist[N+v]+ csrW[i];
      //~ }

      old = minDist[u];
      if(newDist < old){
        minDist[u]=newDist;
        parent[u]=v;
        changed[0]= 1;
      }
    }
    *****/
    // Sh MEM: cp csD
    for(i= start, j=0, minSize=(size<SH_REGS_PER_THREAD?size:SH_REGS_PER_THREAD); i< end && j <minSize ; ++i ,++j)
			shD[ threadIdx.x * SH_REGS_PER_THREAD+j]= csrD[i];
    //******

    //******** 1 PULL SH MEM **********
    for(i = start, j=0; i < end; i++, j++) {        //1 PULL SH MEM
      if( j < SH_REGS_PER_THREAD)
        v = shD[threadIdx.x *SH_REGS_PER_THREAD+j]; //If in ShMem take it
      else
        v = csrD[i];	                              //Else Read from Global
      newDist =  minDist[(id/N)*N+v]+ csrW[i];
      old = minDist[u];
      if(newDist < old){
        minDist[u]=newDist;
        parent[u]=v;
        changed[0]= 1;
      }
    }
    //******** 2 PULL SH MEM **********
    for(i = start, j=0; i < end; i++, j++) {        //1 PULL SH MEM
      if( j < SH_REGS_PER_THREAD)
        v = shD[threadIdx.x *SH_REGS_PER_THREAD+j]; //If in ShMem take it
      else
        v = csrD[i];	                              //Else Read from Global
      newDist =  minDist[(id/N)*N+v]+ csrW[i];
      old = minDist[u];
      if(newDist < old){
        minDist[u]=newDist;
        parent[u]=v;
        changed[0]= 1;
      }
    }
    //******** 3 PULL SH MEM **********
    for(i = start, j=0; i < end; i++, j++) {        //1 PULL SH MEM
      if( j < SH_REGS_PER_THREAD)
        v = shD[threadIdx.x *SH_REGS_PER_THREAD+j]; //If in ShMem take it
      else
        v = csrD[i];	                              //Else Read from Global
      newDist =  minDist[(id/N)*N+v]+ csrW[i];
      old = minDist[u];
      if(newDist < old){
        minDist[u]=newDist;
        parent[u]=v;
        changed[0]= 1;
      }
    }


		/* ~~~~
		if( size < SH_REGS_PER_THREAD){ // adj(u).size() of csrD fits in shared per thread IMPORTATN //24
			// copy csrD and/or csrW to sh
			//~ printf("Copying SH \n");
			for(i= start, j=0; i< end; i++ ,j++)
				 shD[ threadIdx.x * SH_REGS_PER_THREAD + j]= csrD[i];
			// we do not need syncthreads


			// for all its nbrs of id / ith node - PULL	u
			for(i = start, j=0; i< end; i++, j++) { // u and v is adj[u] // Note each nbr is sequential
				//~ v = csrD[i];	//Read
				v = shD[threadIdx.x *SH_REGS_PER_THREAD+ j];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}
			}
			// for all its nbrs of id / ith node - PULL	u
			for(i = start, j=0; i< end; i++, j++){ // u and v is adj[u] // Note each nbr is sequential
				//~ v = csrD[i];	//Read
				v = shD[threadIdx.x*SH_REGS_PER_THREAD + j];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}
			}
			// for all its nbrs of id / ith node - PULL	u
			for(i = start, j=0; i< end; i++, j++){ // u and v is adj[u] // Note each nbr is sequential
				//~ v = csrD[i];	//Read
				v = shD[threadIdx.x *SH_REGS_PER_THREAD+ j];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}
			}

		/*****	// for all its nbrs of id / ith node - PULL	u
			for(i = start, j=0; i< end; i++, j++){ // u and v is adj[u] // Note each nbr is sequential
				//~ v = csrD[i];	//Read
				v = shD[threadIdx.x *SH_REGS_PER_THREAD+ j];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}
			}
			// for all its nbrs of id / ith node - PULL	u
			for(i = start, j=0; i< end; i++, j++){ // u and v is adj[u] // Note each nbr is sequential
				//~ v = csrD[i];	//Read
				v = shD[threadIdx.x *SH_REGS_PER_THREAD+ j];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}
			}
			/// end this
		}
		else {
			//~ printf("NO SH \n");
			// for all its nbrs of id / ith node - PULL	u
			for(i= start; i< end; i++){ // u and v is adj[u] // Note each nbr is sequential
				v = csrD[i];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}

			}
			// for all its nbrs of id / ith node - PULL	u
			for(i= start; i< end; i++){ // u and v is adj[u] // Note each nbr is sequential
				v = csrD[i];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}

			}
			// for all its nbrs of id / ith node - PULL	u
			for(i= start; i< end; i++){ // u and v is adj[u] // Note each nbr is sequential
				v = csrD[i];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}

		}
		/******	// for all its nbrs of id / ith node - PULL	u
			for(i= start; i< end; i++){ // u and v is adj[u] // Note each nbr is sequential
				v = csrD[i];at
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}

			}
			// for all its nbrs of id / ith node - PULL	u
			for(i= start; i< end; i++){ // u and v is adj[u] // Note each nbr is sequential
				v = csrD[i];
				/// relax
				newDist =  minDist[v]+ csrW[i];
				old = minDist[u];
				if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoiding conjection of atomic stmts
					minDist[u]=newDist; //~ old=atomicMin(&minDist[v], newDist); since pull
					parent[u]=v; // do parent later! - done
					changed[0]= 1;
				}

			}
			/// end this
		}
    ~~~~ */
	}
}



void PrintParentOf(int n, int *pArray, int shift, int u, int v, set<pair<int,int > > & stEdges){
	//~ printf("IN printparent %d | %d | (%d_%d)\n", n, shift, u, v);
	int idx = v;
	//~ printf("%d--", idx);
	while(pArray[n*shift+ idx] != -1 ){
		int oldIdx = idx;

		idx = pArray[n*shift+ idx];

		if(oldIdx < idx){
			stEdges.insert(make_pair(oldIdx, idx));
			//~ printf("%d -- %d \n",oldIdx, idx);
		}
		else{
			stEdges.insert(make_pair(idx, oldIdx));
			//~ printf("%d -- %d \n",idx, oldIdx);
		}
	}
	//~ printf("\n");
}


void KMBAlgo(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
bool printEdges = false;
bool printHash  = false;

int main( int argc, char** argv)
{
  if(argc == 1){
    printf("Usage: %s n -p\nn: #SSSPs in parallel. Default n=2\n", argv[0]);
    exit(0);
  }
	if(argc > 1){
		//~ printEdges=true; //prints edges
    sCount = ((atoi(argv[1]) == 0) ?  2 : atoi(argv[1]));
    printHash = true;
	}
  if(argc > 2){
    printEdges = true;
  }
   if(argc > 3){

  }
  //~ if(argc > 1 && strcmp(argv[2],"-p")==0){
      //~ printEdges = true;
  //~ }
  //~ if(argc > 1 && strcmp(argv[3],"-p")==0){
      //~ printEdges = true;
  //~ }
  //~ if(argc > 1 && strcmp(argv[2],"-h")==0){
      //~ printHash = true;
  //~ }
  //~ if(argc > 1 && strcmp(argv[3],"-h")==0){
      //~ printHash = true;
  //~ }

  //~ printf("Scount:%d\n",sCount);

	no_of_nodes=0;
	edge_list_size=0;
  KMBAlgo( argc, argv);

  return EXIT_SUCCESS;
}

int *edges, *edges_wt;
int *h_parentArrays;
int *d_parentArrays;
int N;
set<pair<int,int> > eSet;
set<int> vSet;

void MSTGraph(int t, int *terminals
   , map<pair<int,int>,int> & W
   , set<pair<int,int > > &stEdges
   ,  set<int> &nodeSet
   )
{


  DEBUG printf("in MST Graph1\n");
	N = no_of_nodes;
	DEBUG
	for(int i=0; i < t; i++)
		printf("T %d\n", terminals[i]);

	// for printing  the parent arrays
	DEBUG
	for(int i=0; i < t; i++){
		printf("Parent of %d\n",terminals[i]);
		for(int j=0; j< N; j++){
			printf("P[%d]=%d\n",j,h_parentArrays[j+i*N]);
		}
	}



	//~ printf("In MST\n");


	DEBUG printf("Reading INPUT \n");


	//! DEBUG printf("Reading %d nodes	",no_of_nodes);

   //IMPORTANT CONSTRUCT G' on TE
	no_of_nodes = t;
	edge_list_size = t*(t-1);
	//! DEBUG printf("Reading %d nodes	",no_of_nodes);

  // allocate host memory
	hostMemAllocationNodes();

	// initalize the memory
	for(int i = 0; i < no_of_nodes; i++)
	{
		//~ fscanf(fp,"%d %d",&start,&edgeno);
		//~ scanf("%d %d",&start,&edgeno);
		start = i * (t-1);
		edgeno = t-1;
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		sameindex[i]=i;
		falseval[i]=false;
		trueval[i]=true;
		infinity[i] = INF;
		zero[i]=0;
		h_maxid_maxdegree[i]=-1;

	}

	//read the source node from the file, not needed here though
	//~ scanf("%d",&source);.
	source = 0;

	//~ scanf("%d",&edge_list_size);
	DEBUG
	for(int i=0; i< no_of_nodes;i++)
		printf("V %d: %d,%d\n",i,h_graph_nodes[i].starting,h_graph_nodes[i].no_of_edges);

	DEBUG printf("n=%d 2m=%d \n", no_of_nodes,edge_list_size);

	DEBUG printf("Reading %d edges\n",edge_list_size);


	hostMemAllocationEdges();
	DEBUG printf("BFORE for\n");
	for(int i=0; i < edge_list_size ; i++)
	{


		h_graph_edges[i] = edges[i];
		h_graph_weights[i] = edges_wt[i];

		h_graph_MST_edges[i] = false;
		DEBUG printf("%d: -- %d: %d\n", i,h_graph_edges[i], h_graph_weights[i]);
	}




	//~ DEBUG printf("Finished Reading INPUT\n");

	//~ DEBUG printf("Copying Everything to GPU memory\n");


	//Copy the Node list to device memory

	deviceMemAllocateNodes() ;
	deviceMemAllocateEdges() ;
	deviceMemCopy();

	//~ printf("Before MST\n");

	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
	//~ cudaEventRecord(start);

	GPUMST();

	//~ cudaEventRecord(stop);
	//~ cudaEventSynchronize(stop);
	//~ float milliseconds = 0;
	//~ cudaEventElapsedTime(&milliseconds, start, stop);

	//~ totalTimeMilliSec +=milliseconds;

  DEBUG printf("MST1 Compleet\n");
	cudaMemcpy( test, d_graph_colorindex, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost );
	for(int i=0;i<no_of_nodes;i++)
	{
		 if(test[i]!=0)
		 {
			 printf("1:All Colors not 0, Error at %d\n",i);
			 break;
		 }

	}

	int q=0;
	int minimumCost = 0 ;
	DEBUG printf("Final edges present in MST\n");
	cudaMemcpy( h_graph_MST_edges, d_graph_MST_edges, sizeof(bool)*edge_list_size, cudaMemcpyDeviceToHost) ;


	int v1 = 0; //IMP to

	for(int i=0;i<int(edge_list_size);i++)
	{
		int v1Limit = h_graph_nodes[v1].starting +h_graph_nodes[v1].no_of_edges ;
		if(i == v1Limit) // if limit reached, move to next v1
			v1++;

		//~ printf("%d :",i);
		if(h_graph_MST_edges[i])
		{
			int v2 = h_graph_edges[i];
			int edgeweight = h_graph_weights[i];
			//~ DEBUG printf("%d %d %d\n",  v1 + 1, v2 + 1, edgeweight); // For Checking with CPU
			//int u = terminals[v1];
			//int v = terminals[v2];
			//~ printf("%d -- %d: %d == %d -- %d: %d\n", v1, v2, edgeweight, u, v, edgeweight);
			//printf("%d -+- %d \n",  u, v );

			vSet.insert(v1);
			vSet.insert(v2);
			if(v1 < v2)
				eSet.insert(make_pair(v1,v2));
			else
				eSet.insert(make_pair(v2,v1));

			minimumCost += edgeweight;
			q++;
		}
		//~ printf("\n");
		// Post increment after printing!
	}
	// For each terminal on their respective parent array!



	DEBUG printf("Printing Parent array\n");

	for (std::set<pair< int,int> >::iterator it=eSet.begin(), end=eSet.end(); it!= end; ++it){
		int v1= it->first ; int u = terminals[v1];
		int v2= it->second; int v = terminals[v2];

		//~ printf("%d -- %d\n", u,v);
		PrintParentOf(N,h_parentArrays,v1, u,v, stEdges); // populates stEdges
	}

	unsigned mstVal= 0;

	for (std::set<pair< int,int> >::iterator it=stEdges.begin(), end=stEdges.end(); it!= end; ++it){
		int v1= it->first ;
		int v2= it->second;
		mstVal += W[make_pair(v1,v2)];
		//! if(printEdges)
			//! printf("%d  -- %d\n",v1+1, v2+1);
			//~ printf("%d %d %d\n",v1+1, v2+1, W[make_pair(v1,v2)]);
      nodeSet.insert(v1);
      nodeSet.insert(v2);
	}


 	//~ printf("No of edges in MST: %d, no of nodes: %d cost %d\n",q,no_of_nodes,minimumCost);
	//~ printf("VALUE %d\n",minimumCost);
	//~ printf("%f\n",totalTimeMilliSec*1000); //ms to s
  //! if(!printEdges)
  //printf("Node == Node TMP? %d==%d\n", (int)nodeSet.size() , (int)nodeSetTemp.size());

  if(stEdges.size() == nodeSet.size()-1) { // MST(G') is tree alread then we do not have to do G"
    cudaEventRecord(tstop);
		cudaEventSynchronize(tstop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, tstart, tstop);
		totalTimeMilliSec +=milliseconds;

    if(printEdges){
      for (std::set<pair< int,int> >::iterator it=stEdges.begin(), end=stEdges.end(); it!= end; ++it){
        int v1= it->first ;
        int v2= it->second;
        printf("%d %d\n",v1+1, v2+1);
      }
    }
    printf("VALUE %d,%f\n",mstVal, totalTimeMilliSec);
  }




	//! freeMem(); // This frees the GPU memory as well! OMG!
  DEBUG printf("in MST Graph1\n");
}

// Construct G" and MST(G")
void MSTGraphG2(set<pair<int,int>>  &stEdges, set<int> &nodeSet, map<pair<int,int>,int> & W)
{
  DEBUG printf("in MST Graph2\n");
  N = nodeSet.size(); //need to be modified.
  //! printf("N=%d\n", N);
  vector< vector<int> > graph(N);

   unordered_map <int,int> vMap;
   vector<int> nodeVec(nodeSet.begin(), nodeSet.end());
   int i=0;
   for(auto &a: nodeVec){
      //! printf("= %d \n", a);
      vMap[a]=i++;
   }

   for(auto &a: stEdges){
      int v1 = vMap[a.first];
      int v2 = vMap[a.second];
      //! printf("%d --- %d \n" ,v1,v2 );
      graph[v2].push_back(v1);
      graph[v1].push_back(v2);
      //! nodeSet.insert(a.first);
      //! nodeSet.insert(a.second);
   }



	//~ printf("In MST\n");
	int eSize = stEdges.size() * 2;

	DEBUG printf("Reading INPUT \n");
	//~ scanf("%d",&no_of_nodes);


	// allocate host memory
	DEBUG printf("Reading %d nodes	",no_of_nodes);

   //IMPORTANT
	no_of_nodes = N;  // n
	edge_list_size = eSize; // 2m

	hostMemAllocationNodes();
	int cumSum = 0;
	// initalize the memory
	for(int i = 0; i < no_of_nodes; i++)
	{
		//~ fscanf(fp,"%d %d",&start,&edgeno);
		//~ scanf("%d %d",&start,&edgeno);
		start = cumSum;      // start of csr(i) for i \in V
		auto adjSize = graph[i].size();
		cumSum += adjSize;
		edgeno = adjSize;           // |N(i)|
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		sameindex[i]=i;         // i // this is good!
		falseval[i]=false;
		trueval[i]=true;
		infinity[i] = INF;
		zero[i]=0;
		h_maxid_maxdegree[i]=-1;

	}

	//read the source node from the file, not needed here though
	//~ scanf("%d",&source);.
	source = 0;

	//~ scanf("%d",&edge_list_size);
	DEBUG
	for(int i=0; i< no_of_nodes;i++)
		printf("V %d: %d,%d\n",i,h_graph_nodes[i].starting,h_graph_nodes[i].no_of_edges);

	DEBUG printf("n=%d 2m=%d \n", no_of_nodes,edge_list_size);

	DEBUG printf("Reading %d edges\n",edge_list_size);

	//~ int id,cost;

	hostMemAllocationEdges();
	DEBUG printf("BFORE for\n");
	/*
	for(int i=0; i < edge_list_size ; i++)
	{
		//~ scanf("%d",&id);
		//~ scanf("%d",&cost);
		h_graph_edges[i] = edges[i];
		h_graph_weights[i] = edges_wt[i];
		h_graph_MST_edges[i] = false;
		DEBUG printf("%d: -- %d: %d\n", i,h_graph_edges[i], h_graph_weights[i]);
	}
	*/
	int u = 0;
  i = 0;
  for (auto adjList : graph){
    int v1 = nodeVec[u];    // Thanks Rupesh!

    for (auto v : adjList){
      int v2 = nodeVec[v];

      //! printf(" %d %d: %d\n",v1,v2 , W[{v1,v2}]);

      h_graph_edges[i] = v;
      h_graph_weights[i] = W[{v1,v2}];
      h_graph_MST_edges[i] = false;

      ++i;
    }
    ++u;
	}


	//~ DEBUG printf("Finished Reading INPUT\n");
	//~ DEBUG printf("Copying Everything to GPU memory\n");


	//~ Copy the Node list to device memory

	deviceMemAllocateNodes() ;
	deviceMemAllocateEdges() ;
	deviceMemCopy();

	//~ printf("Before MST\n");

	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
	//~ cudaEventRecord(start);

	GPUMST();

	//~ cudaEventRecord(stop);
	//~ cudaEventSynchronize(stop);
	//~ float milliseconds = 0;
	//~ cudaEventElapsedTime(&milliseconds, start, stop);

	//~ totalTimeMilliSec +=milliseconds;


	cudaMemcpy( test, d_graph_colorindex, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost );
	for(int i=0;i<no_of_nodes;i++)
	{
		 if(test[i]!=0)
		 {
			 printf("2:All Colors not 0, Error at %d\n",i);
			 break;
		 }

	}

	int q=0;
	int minimumCost = 0 ;
	DEBUG printf("Final edges present in MST\n");
	cudaMemcpy( h_graph_MST_edges, d_graph_MST_edges, sizeof(bool)*edge_list_size, cudaMemcpyDeviceToHost) ;

  cudaEventRecord(tstop);
  cudaEventSynchronize(tstop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, tstart, tstop);
  totalTimeMilliSec +=milliseconds;


	int v1 = 0; //IMP to
	eSet.clear(); //reset!
	for(int i=0;i<int(edge_list_size);++i)
	{
		int v1Limit = h_graph_nodes[v1].starting +h_graph_nodes[v1].no_of_edges ;
		if(i == v1Limit) // if limit reached, move to next v1
			v1++;

		//~ printf("%d :",i);
		if(h_graph_MST_edges[i])
		{
			int v2 = h_graph_edges[i];
			int edgeweight = h_graph_weights[i];
			//~ DEBUG printf("%d %d %d\n",  v1 + 1, v2 + 1, edgeweight); // For Checking with CPU

			int u = nodeVec[v1] + 1; //for printing
			int v = nodeVec[v2] + 1;

			//! //~ printf("%d -- %d: %d == %d -- %d: %d\n", v1, v2, edgeweight, u, v, edgeweight);
			if(printEdges)
				printf("%d %d \n",  u, v );

			//! vSet.insert(v1);
			//! vSet.insert(v2);
			//! if(v1 < v2)
				//! eSet.insert(make_pair(v1,v2));
			//! else
				//! eSet.insert(make_pair(v2,v1));  // assuming no duplicates

			minimumCost += edgeweight;
			q++;
		}
		//~ printf("\n");
		// Post increment after printing!
	}
	// For each terminal on their respective parent array!

	//! printf("%d,%f, %f\n",minimumCost, totalTimeMilliSec, milliseconds);
	printf("VALUE %d,%f, %f\n",minimumCost, totalTimeMilliSec, milliseconds);

	//! DEBUG printf("Printing Parent array\n");

	//! for (std::set<pair< int,int> >::iterator it=eSet.begin(), end=eSet.end(); it!= end; ++it){
		//! int v1= it->first ; int u = nodeVec[v1];
		//! int v2= it->second; int v = nodeVec[v2];

		//! //~ printf("%d -- %d\n", u,v);
	//! }

	//! unsigned mstVsal= 0;


	//! for (std::set<pair< int,int> >::iterator it=stEdges.begin(), end=stEdges.end(); it!= end; ++it){

		//! int v1= it->first ;
		//! int v2= it->second;
		//! mstVal += W[make_pair(v1,v2)];
		//! if(printEdges)
			//! printf("%d %d\n",v1+1, v2+1);
			//! //~ printf("%d %d %d\n",v1+1, v2+1, W[make_pair(v1,v2)]);
	//! }

 	//~ printf("No of edges in MST: %d, no of nodes: %d cost %d\n",q,no_of_nodes,minimumCost);
	//~ printf("VALUE %d\n",minimumCost);
	//~ printf("%f\n",totalTimeMilliSec*1000); //ms to s
  DEBUG printf("in MST Graph2\n");
	freeMem();

}


////////////////////////////////////////////////////////////////////////////////
//Apply Shortest Path on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////

void KMBAlgo( int argc, char** argv)
{

  DEBUG printf("Using sCount:%d\n",sCount);
    //cudaSetDevice(1);
	size_t mf, ma;
	cudaError_t err = cudaMemGetInfo(&mf, &ma);
	if(err != cudaSuccess)
		printf("ALERT: %s \n",cudaGetErrorString(err));

	//~ int source=0;
	//~ int source2=0;
  int* source = (int*) malloc(sizeof(int)*(sCount));

	scanf("%d",&no_of_nodes);
	DEBUG printf("|V|: %d\n",no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK) {
    num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK);
    num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	// initalize the memory
	// allocate host memory
  int* h_graph_nodes = (int*) malloc(sizeof(int)*(no_of_nodes+1)); //  +1 for csrM



  int start, edgeno;

	int no=0;
  for( unsigned int i = 0; i < no_of_nodes; i++)  {
    scanf("%d %d",&start,&edgeno);
    DEBUG printf("%d %d\n", start,edgeno);
    if(edgeno>100)
      no++;
    h_graph_nodes[i] = start;
  }

  h_graph_nodes[no_of_nodes]=start+edgeno;

  //read the source int from the file
  //~ fscanf(fp,"%d",&source);
  //~ scanf("%d",&source);
  int dummy;
  scanf("%d",&dummy); // not use else where

  //~ DEBUG for(int ii=0; ii < sCount; ++ii)
    //~ printf("src[%d]: %d\n",ii,source[ii]);
  //set the source int as true in the mask

  //! h_graph_mask[source]=true;
  //h_graph_counter[source]=0;

  //~ fscanf(fp,"%d",&edge_list_size);
  scanf("%d",&edge_list_size);
  //~ printf("|E|: %d\n", edge_list_size);

  int* h_graph_edges    = (int*) malloc(sizeof(int)*edge_list_size);
	int* h_graph_weights  = (int*) malloc(sizeof(int)*edge_list_size);

	map<pair<int,int> , int> W;

	int id;

  for(int i=0, j = 0; i < edge_list_size ; i++)
  {
    int id1;
    if(i >= h_graph_nodes[j+1]) j++;
    scanf("%d %d",&id, &id1);
    h_graph_edges[i] = id;
    h_graph_weights[i] = id1;
    DEBUG printf("%d %d\n",h_graph_edges[i],h_graph_weights[i] );

    W[make_pair(j,id)] = id1;
    W[make_pair(id, j)] = id1;
    //~ printf("(%d, %d) -- %d\n", j, id, W[make_pair(j,id)]);
  }

  int terminalSize;
  //bool isOddTerminals =false;
  scanf("%d", &terminalSize);

  //~ if(terminalSize % 2 == 1)
    //~ isOddTerminals = true;

  int terminals[terminalSize];

  for(int i=0; i < terminalSize; i++){
    scanf("%d", &id);
		terminals[i] = id;
    //! h_isTerminal[id]=true; //Not in use
		//~ printf("T\t%d\n",id);
	}



	h_parentArrays = (int*) malloc(sizeof(int)*(      no_of_nodes*terminalSize));
  cudaMalloc( (void**) &d_parentArrays, sizeof(int)*no_of_nodes*terminalSize);

	int edgeId = 0;
	//~ printf("N=%d, 2M=%d\n", no_of_nodes, edge_list_size);

	edges     = (int*) malloc(sizeof(int)*terminalSize*(terminalSize-1)); //For MST
	edges_wt  = (int*) malloc(sizeof(int)*terminalSize*(terminalSize-1));


	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);



	//Copy the int list to device memory
	int* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(int)*(no_of_nodes+1)) ;  //+1 for csrM

	//Copy the Edge List to device Memory
	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;

	int* d_graph_weights;
	cudaMalloc( (void**) &d_graph_weights, sizeof(int)*edge_list_size);



	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes * sCount); //Rupesh
	//! for(int i=0;i<no_of_nodes;i++) //happen  in kernelInit
		//! h_cost[i]= MAX_COST;


	// allocate device memory for result / OUTPUT
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes * sCount);
	//~ cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

  int *h_parent = (int*) malloc(sizeof(int)*no_of_nodes *sCount);

	// copy the parent array
	int* d_parent;
	cudaMalloc( (void**) &d_parent, sizeof(int)*no_of_nodes * sCount) ;

  bool *d_changed;
	bool *changed = (bool*)   malloc(sizeof(bool));
	cudaMalloc( (void**) &d_changed, sizeof(bool));
	cudaCheckError();

  // new for kSSSP
  int* d_sources;
	cudaMalloc( (void**) &d_sources, sizeof(int) * sCount) ;
  //cudaMemcpy( d_sources, source, (sizeof(int)*(sCount)), cudaMemcpyHostToDevice) ;
	cudaCheckError();

  /*************
   * TIMER CODE
   *************/
  cudaEventCreate(&tstart);
  cudaEventCreate(&tstop);
  cudaEventRecord(tstart);
  cudaCheckError();


	cudaMemcpy( d_graph_nodes, h_graph_nodes, (sizeof(int)*(no_of_nodes+1)), cudaMemcpyHostToDevice) ; // +1 for csrM
	cudaCheckError();

	cudaMemcpy( d_graph_weights, h_graph_weights, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);
	cudaCheckError();


  int tempScount = sCount; // Just to ensure the last run runs < sCount times
  printf("sCount:%d terminalSize:%d n:%d m:%d\n",sCount, terminalSize,no_of_nodes, edge_list_size);

	for(int it=0, end=(terminalSize+sCount-1)/sCount; it < end; ++it){ // ceil(terminalSize/sCount)

    if(terminalSize%sCount==0 || it!=end-1){// Thanks Rupesh. termSize%sCount==0 || it!=end-1
      tempScount= sCount;
      for(int ii=0; ii < sCount; ++ii ){
        source[ii]  = terminals[sCount*it+ii];
        DEBUG printf("\t #%d SSSP from %d\n", ii+1, source[ii] );
      }
      num_of_blocks = (sCount*no_of_nodes + MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK;

    }else{
      tempScount = terminalSize%sCount; // suposedly 1 t0 sCount-1 for the last round when sCount does not divide terminalSize
      for(int ii=0, endII=tempScount; ii < endII ; ++ii ){
        source[ii]  = terminals[sCount*it+ii];
        DEBUG printf("\t #%d SSSP from %d\n", ii+1, source[ii] );
      }

      num_of_blocks = (tempScount*no_of_nodes + MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK;
    }

    num_of_threads_per_block = MAX_THREADS_PER_BLOCK;//(no_of_nodes<MAX_THREADS_PER_BLOCK? no_of_nodes: MAX_THREADS_PER_BLOCK);
    dim3  gridKN( num_of_blocks, 1, 1);
    dim3  threadsKN( num_of_threads_per_block, 1, 1);


		int k=0;
    /*************
     * TIMER CODE
     *************/
		//~ cudaEventCreate(&tstart);
		//~ cudaEventCreate(&tstop);
		//~ cudaEventRecord(tstart);
		//~ cudaCheckError();

    cudaMemcpy( d_sources, source, (sizeof(int)*(tempScount)), cudaMemcpyHostToDevice) ;        //
    kernelInitDistAndParent<<< gridKN , threadsKN >>>( no_of_nodes, d_cost,d_parent, tempScount); //SAME grid
    kernelInitSources<<< 1 , 1 >>>(no_of_nodes,d_sources, d_cost, tempScount); //

    cudaCheckError();

    do{
      changed[0] = false;

      cudaMemcpy( d_changed, changed, sizeof(bool), cudaMemcpyHostToDevice);

      cudaCheckError();


      csrKernelBellmanFordMoore<<< gridKN, threadsKN  >>>( no_of_nodes, d_sources ,
        d_graph_nodes, d_graph_edges, d_graph_weights,  // inputs
        d_changed, // fixed pt var
        d_cost, d_parent,  //these are outputs
        sCount,
        tempScount);

      cudaCheckError();

      cudaMemcpy( changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
      cudaCheckError();

      k++;

      DEBUG printf("%d -- FINSHED? %s\n",k, (!changed[0]? "Yes": "No"));
    }while(changed[0] == true);
    DEBUG printf("AFTER LAUNCH\n");
		//~ cudaDeviceSynchronize();


		//~ cudaEventRecord(tstop);
		//~ cudaEventSynchronize(tstop);
		//~ float milliseconds = 0;
		//~ cudaEventElapsedTime(&milliseconds, tstart, tstop);
		//~ totalTimeMilliSec +=milliseconds;

		DEBUG printf("\nTOTAL IT:%d\n",k);

    cpyParentArrayNew<<<gridKN, threadsKN>>>(no_of_nodes, it, d_parentArrays, d_parent, sCount, tempScount);        //~DOUBLE~  K COPY
    cudaCheckError();

    // copy result from device to host
    cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes*tempScount, cudaMemcpyDeviceToHost);      //~DOUBLE~  K COPY

    unsigned long long int sol;

    DEBUG printf("N=%d %d\n",no_of_nodes, INT_MAX/2);
    //~ bool flagWarn = false;
    DEBUG
    for(int jj=0; jj < tempScount; ++jj){
      sol = 0;
      for(int i=0; i < no_of_nodes; ++i){
        //~ if(h_cost[jj*no_of_nodes+i] < INT_MAX/2)
          sol += h_cost[jj*no_of_nodes+i];
        //~ else if(!flagWarn){
          //~ printf("WARN: %d \n", h_cost[jj*no_of_nodes+i]);
          //~ flagWarn = true;
        //~ }
        //if(printEdges) printf("d1[%d]=%d  p1[%d]=%d\n",i,h_cost[i+jj*no_of_nodes], i, h_parent[i+jj*no_of_nodes]);
      }
      if(printHash) printf("iterat:%d SSSP %d on src %d HASH VAL %lld\n", it, jj+1, source[jj], sol);
    }

    //! return;


    //! int pt1 = terminals[it];
		//! for(int j=0; j<terminalSize; j++){
			//! int pt2 = terminals[j];
			//! if(pt1 != pt2){
				//! DEBUG printf("%d -- %d: %d  \n", pt1+1 ,pt2+1 , h_cost[pt2]);
				//! DEBUG printf("E[%d] %d\n", edgeId, j);
				//! edges[edgeId] = j;
				//! edges_wt[edgeId] = h_cost[pt2];
				//! edgeId++;
			//! }
		//! }

      for(int ii=0; ii < tempScount; ++ii){
        int pt1 = terminals[sCount*it + ii ];
        for(int j=0; j<terminalSize; ++j){
          int pt2 = terminals[j];
          if(pt1 != pt2){
            edges[edgeId] = j;
            edges_wt[edgeId] = h_cost[no_of_nodes * ii + pt2]; ///Mod
            edgeId++;
          }
        }
      }
	}

  //return;
	cudaMemcpy( h_parentArrays, d_parentArrays, sizeof(int)*no_of_nodes*terminalSize, cudaMemcpyDeviceToHost); // why is this needed? It is used inside MST1
  ///printf("SSSP DONE\n");
	DEBUG printf("GPU [ms]:%f\n",totalTimeMilliSec);

	// Construct G' and Launch the kernel for the MST(G')
	set<pair<int,int > > stEdges ;
  set<int> nodeSet;
  DEBUG printf("In main before MST\n");
  MSTGraph(terminalSize,terminals, W, stEdges, nodeSet);

  // Construct G" and Launch the kernel for the MST(G")
  if(stEdges.size() != nodeSet.size()-1)  //|E| != |V|-1
    MSTGraphG2(stEdges, nodeSet, W);
  //~ else
   //~ printf("Tree already\n");

  //~ cudaDeviceSynchronize();

  free( h_graph_nodes);
  free( h_graph_edges);
  free( h_graph_weights);
  free( h_cost);

	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);

	cudaFree(d_graph_weights);

	cudaFree(d_cost);
	cudaFree(d_parentArrays);

}
