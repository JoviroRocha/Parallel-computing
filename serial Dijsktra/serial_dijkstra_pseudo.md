func InitSSSP(origin):
    dist(origin) ← 0
    pred(origin) ← Null
        for all vertices v != origin
            dist(v) ← ∞
            pred(v) ← Null

Relax(u->v):
    dist(v) ← dist(u) + w(u->v)
    pred(v) ← u

func Dijkstra(origin):
    InitSSSP(origin)
    Insert(origin, 0)
    while the priority queue is not empty
        u ← ExtractMin( ) //
        for all edges u -> v
            if u -> v is tense
                Relax(u->v)
                if v is in the priority queue
                    DecreaseKey(v, dist(v))
                else
                    Insert(v, dist(v))

// References:
// 1- https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf [ Dijkstra ]
// 2- https://www.geeksforgeeks.org/priority-queue-using-binary-heap/ [ Binary Heap ]