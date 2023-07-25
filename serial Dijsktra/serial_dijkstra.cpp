#include <bits/stdc++.h>
using namespace std;

struct Vertex {
    
    int id; // or string, IDK
    int weight;
 
};

Vertex Heap[1000];
int size = -1;

// Dijkstra implementation 
// ...

// Functions for the binary heap

int parent(int i)
{
    return (i - 1) / 2;
}

int leftChild(int i)
{
 
    return ((2 * i) + 1);
}

int rightChild(int i)
{
 
    return ((2 * i) + 2);
}

void shiftUp(int i)
{
    while (i > 0 && H[parent(i)].weight > H[i].weight) {

        swap(H[parent(i)], H[i]);
 
        i = parent(i);
    }
}

void shiftDown(int i)
{
    int minIndex = i;
 
    int l = leftChild(i);
    if (l <= size && H[l].weight < H[minIndex].weight) {
        minIndex = l;
    }
 
    int r = rightChild(i);
    if (r <= size && H[r].weight < H[minIndex].weight) {
        minIndex = r;
    }
 
    if (i != minIndex) {
        swap(H[i], H[minIndex]);
        shiftDown(minIndex);
    }
}

void insert(Vertex p)
{
    size = size + 1;
    H[size] = p;
 
    shiftUp(size);
}

Vertex extractMin()
{
    Vertex result = H[0];

    H[0] = H[size];
    size = size - 1;
 
    shiftDown(0);
    return result;
}

void changePriority(int i, int newWeight)
{
    Heap[i].weight = newWeight;
    shiftUp(i)
}