#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

struct Vertex{
   int weight;
   int vertex;
};

struct PriorityQueue
{
private:
    vector<Vertex> A;
 
    int PARENT(int i) {
        return (i - 1) / 2;
    }
 
    int LEFT(int i) {
        return (2*i + 1);
    }
 
    int RIGHT(int i) {
        return (2*i + 2);
    }
 
    void heapify_down(int i)
    {
        // obtém filho esquerdo e direito do nó no índice `i`
        int left = LEFT(i);
        int right = RIGHT(i);
 
        int smallest = i;
 
        // compara `A[i]` com seu filho esquerdo e direito
        // e encontrar o menor valor
        if (left < size() && A[left].weight < A[i].weight) {
            smallest = left;
        }
 
        if (right < size() && A[right].weight < A[smallest].weight) {
            smallest = right;
        }
 
        // troca com um filho de menor valor e
        // chama heapify-down no filho
        if (smallest != i)
        {
            swap(A[i], A[smallest]);
            heapify_down(smallest);
        }
    }
 
    void heapify_up(int i)
    {
        if (i && A[PARENT(i)].weight > A[i].weight)
        {
            swap(A[i], A[PARENT(i)]);
 
            heapify_up(PARENT(i));
        }
    }
 
public:
    // retorna o tamanho do heap
    unsigned int size() {
        return A.size();
    }

    void print(){

        for(auto i = A.begin(); i < A.end(); i++){
            printf("%d %d  ", A[i - A.begin()].vertex, A[i - A.begin()].weight);
        }
        printf("\n");
    }
    
    void update(Vertex update_vertex){

        A.erase( remove_if(A.begin(), A.end(), [&](Vertex const & vertex) {return update_vertex.vertex == vertex.vertex;}), A.end());
        
    }


    // Função para verificar se o heap está vazio ou não
    bool empty() {
        return size() == 0;
    }
 
    // insere a chave no heap
    void push(int weight, int vertex)
    {
        // insere um novo elemento no final do vector
        A.push_back(Vertex{weight=weight, vertex=vertex});
 
        // obtém o índice do elemento e chama o procedimento heapify-up
        int index = size() - 1;
        heapify_up(index);
    }
 
    // Função para remover um elemento de menor prioridade (presente na raiz)
    void pop()
    {
        try {
            // se o heap não tiver elementos, lança uma exceção
            if (size() == 0)
            {
                throw out_of_range("Vector<X>::at() : "
                        "index is out of range(Heap underflow)");
            }
 
            // substitui a raiz do heap pelo último elemento
            // do vector
            A[0] = A.back();
            A.pop_back();
 
            // chama heapify-down no nó raiz
            heapify_down(0);
        }
        // captura e imprime a exceção
        catch (const out_of_range &oor) {
            cout << endl << oor.what();
        }
    }
 
    // Função para retornar um elemento com a prioridade mais baixa (presente na raiz)
    Vertex top()
    {
        try {
            // se o heap não tiver elementos, lança uma exceção
            if (size() == 0)
            {
                throw out_of_range("Vector<X>::at() : "
                        "index is out of range(Heap underflow)");
            }
 
            // caso contrário, retorna o elemento superior (primeiro)
            return A.at(0);        // ou return A[0];
        }
        // captura e imprime a exceção
        catch (const out_of_range &oor) {
            cout << endl << oor.what();
        }
    }
};

int main() {

    string file_name;
    int amount_vertex;
    fstream new_file;

    cout << "Enter the quantity of vertex:\n";
    cin >> amount_vertex;
    cout << "Enter the file path:\n";
    cin >> file_name;

    // Create a linear matrix with the weights
    auto __size = amount_vertex * amount_vertex + 1;
    auto linear_matrix = (int *) malloc(__size * sizeof(int));
    for (int i = 0; i < __size; i++){
        linear_matrix[i] = -1;
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
            linear_matrix[position] = stoi(vertex_3);
        }
        new_file.close(); 
    }

    // initSSP
    int result[amount_vertex];
    PriorityQueue dist;
    dist.push(0, 1);
    result[0] = 0;
    for (int i = 2; i <= amount_vertex; i++){
        dist.push(INT_MAX, i);
        result[i - 1] = INT_MAX;
    }

    // Dijkstra
    while(!dist.empty()){
        Vertex minor = dist.top();
        dist.pop();
        if(minor.weight == INT_MAX){
            result[minor.vertex - 1] = 0;
            continue;
        }
        for(int i = 2; i <= amount_vertex; i++){
            int position = (minor.vertex - 1) * amount_vertex + i;
            int weight_edge = linear_matrix[position];
            if(weight_edge == -1){
                continue;
            }
            int minor_weight = result[minor.vertex - 1];
            int new_weight = weight_edge + minor_weight;
            if(result[i - 1] > new_weight){
                result[ i - 1 ] = new_weight;
                dist.update(Vertex{new_weight, i});
                dist.push(new_weight, i);
            }
        }
    }

    for(int i = 0; i < amount_vertex; i++){
            printf("%d: %d\n", i + 1, result[i]);
    }
    

    return 0;
}