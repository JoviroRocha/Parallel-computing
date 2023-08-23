%%writefile parallel.cu
#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

const size_t threadsPerBlock = 256;

__global__ void dijkstra_update(int *result, int *visited, int *device_linear_matrix, int amount_vertex){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; index < amount_vertex; index += stride){

        // stop execution if index was visited
        if(visited[index - 1] == 1 || index == 0) continue;

        for(int loop = 1; loop < amount_vertex; loop++){
            if(visited[loop - 1] == 1 && loop != index){
            
                int position = (loop - 1) * amount_vertex + index;// Can change the index and the loop and remove -1
                int edgeweight = device_linear_matrix[position];
                if(edgeweight == -1 || result[loop - 1] == INT_MAX -1){
                    continue;
                }
                int newweight = result[loop - 1] + edgeweight;
                if (newweight < result[index - 1]){
                    result[index - 1] = newweight;
                }

            }
        }
    }
}

__global__ void dijkstra_minor(int* result, int* visited, int n) {
   int minor = INT_MAX;
   int index_min;
    for( int x = 0; x < n; x++){
        if(result[x] < minor && visited[x] == 0){
            minor = result[x];
            index_min = x;
        }
   }
   visited[index_min] = 1;
}

__global__ void initSSP(int *result, int *visited, int amount_vertex){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; index < amount_vertex; index += stride){
        if(index == 0){
            result[index] = 0;
        } else{
            result[index] = INT_MAX - 1;
        }
        visited[index] = 0;
    }
}

int main() {
    // Cuda Setup
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    const size_t blocksPerGrid =  prop.multiProcessorCount * 32;

    // Get Data about the graph
    string file_name;
    int amount_vertex;
    fstream new_file;

    cout << "Enter the quantity of vertex:\n";
    cin >> amount_vertex;
    cout << "Enter the file path:\n";
    cin >> file_name;

    // Create a linear matrix with the weights
    auto __size = amount_vertex * amount_vertex + 1;
    int *host_linear_matrix;
    int *device_linear_matrix;

    cudaMalloc((void **)&device_linear_matrix, __size * sizeof(int));
    host_linear_matrix = new int[__size];

    for (int i = 0; i < __size; i++){
        host_linear_matrix[i] = -1;
    }
    new_file.open(file_name, ios::in); 
    
    if (new_file.is_open()) { 
        string sa;
        while (getline(new_file, sa)) {
            istringstream iss(sa);
            string vertex_1;
            string vertex_2;
            string vertex_3;
            getline(iss, vertex_1, ' ');
            getline(iss, vertex_2, ' ');
            getline(iss, vertex_3, ' ');

            int position = (stoll(vertex_1) - 1) * amount_vertex + stoll(vertex_2);
            host_linear_matrix[position] = stoi(vertex_3);
        }
        new_file.close(); 
    }

    cudaMemcpy(device_linear_matrix, host_linear_matrix, __size * sizeof(int), cudaMemcpyHostToDevice);
    delete [] host_linear_matrix;
   
    // Declare some dijkstra variables:
    int * result;
    int * visited;
    cudaMalloc((void **)&result, amount_vertex * sizeof(int));
    cudaMalloc((void **)&visited, amount_vertex * sizeof(int));

    //InitSSP
    initSSP<<<blocksPerGrid, threadsPerBlock>>>(result, visited, amount_vertex);
    cudaDeviceSynchronize();

    //Dijkstra
    for(int i = 0; i < amount_vertex; i++){
        //find minor
        dijkstra_minor<<<1, 1>>>(result, visited, amount_vertex);
        cudaDeviceSynchronize();

        // update minor brothers
        dijkstra_update<<<blocksPerGrid, threadsPerBlock>>>(result, visited, device_linear_matrix, amount_vertex);
        cudaDeviceSynchronize();
    }

    int *res = new int[amount_vertex];
    cudaMemcpy(res, result, amount_vertex * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < amount_vertex; i++){
        printf("%d: %d\n", i, res[i]);
    }
    //Free memory
    delete [] res;
    cudaFree(visited);
    cudaFree(device_linear_matrix);
    return 0;
}