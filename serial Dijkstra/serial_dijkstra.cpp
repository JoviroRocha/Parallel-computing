#include <bits/stdc++.h>
using namespace std;

struct Vertex {
    
    int id; // or string, IDK
    int weight;
    Vertex * previous;
 
};

Vertex Heap[1000];
int vertices_amount = -1;

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
    while (i > 0 && Heap[parent(i)].weight > Heap[i].weight) {

        swap(Heap[parent(i)], Heap[i]);
 
        i = parent(i);
    }
}

void shiftDown(int i)
{
    int minIndex = i;
 
    int l = leftChild(i);
    if (l <= vertices_amount && Heap[l].weight < Heap[minIndex].weight) {
        minIndex = l;
    }
 
    int r = rightChild(i);
    if (r <= vertices_amount && Heap[r].weight < Heap[minIndex].weight) {
        minIndex = r;
    }
 
    if (i != minIndex) {
        swap(Heap[i], Heap[minIndex]);
        shiftDown(minIndex);
    }
}

void insert(Vertex p)
{
    vertices_amount = vertices_amount + 1;
    Heap[vertices_amount] = p;
 
    shiftUp(vertices_amount);
}

Vertex extractMin()
{
    Vertex result = Heap[0];

    Heap[0] = Heap[vertices_amount];
    vertices_amount = vertices_amount - 1;
 
    shiftDown(0);
    return result;
}

void changePriority(int i, int newWeight)
{
    Heap[i].weight = newWeight;
    shiftUp(i);
}