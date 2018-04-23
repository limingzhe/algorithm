package DataStructure;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

/*
最短路径
(1)当权值为非负时，用Dijkstra。
(2)当权值有负值，且没有负圈，则用SPFA，SPFA能检测负圈，但是不能输出负圈。
(3)当权值有负值，而且可能存在负圈，则用BellmanFord，能够检测并输出负圈。
(4)SPFA检测负环：当存在一个点入队大于等于V次，则有负环，后面有证明。
 */
public class DirectedMap {
    private char[] V;   // 顶点
    private int[][] E;  // 边
    private int EdgeNum;    // 边的数量
    private static final int INF = Integer.MAX_VALUE;   // 最大值

    public DirectedMap(char[] vexs, int[][] edges) {
        V = new char[vexs.length];
        System.arraycopy(vexs, 0, V, 0, vexs.length);

        E = new int[vexs.length][vexs.length];
        for (int i = 0; i < edges.length; i++) {
            for (int j = 0; j < edges[0].length; j++) {
                E[i][j] = edges[i][j];
            }
        }

        EdgeNum = 0;
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < V.length; j++) {
                if (E[i][j] != Integer.MAX_VALUE) EdgeNum++;
            }
        }
    }

    // 得到c的下标
    private int getPosition(char c) {
        for (int i = 0; i < V.length; i++) {
            if (V[i] == c) return i;
        }
        return -1;
    }

    public void print() {
        System.out.println("Matrix Graph:");
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < V.length; j++) {
                System.out.print(E[i][j] + " ");
            }
            System.out.print('\n');
        }
    }

    public void DFS(char start) {
        boolean[] visited = new boolean[V.length];
        for (int i = 0; i < V.length; i++) {
            visited[i] = false;
        }

        Stack<Character> stack = new Stack();
        stack.push(start);
        visited[getPosition(start)] = true;

        while (!stack.isEmpty()) {
            Character c = stack.pop();
            System.out.print(c);
            for (int i = 0; i < E[0].length; i++) {
                if (E[getPosition(c)][i] != INF && visited[i] == false) {
                    stack.push(V[i]);
                    visited[i] = true;
                }
            }
        }
        System.out.print('\n');
    }

    public void BFS(char start) {
        boolean[] visited = new boolean[V.length];
        for (int i = 0; i < V.length; i++) {
            visited[i] = false;
        }

        Queue<Character> queue = new LinkedList<>();
        queue.offer(start);
        visited[getPosition(start)] = true;

        while (!queue.isEmpty()) {
            Character c = queue.poll();
            System.out.print(c);
            for (int i = 0; i < E[0].length; i++) {
                if (E[getPosition(c)][i] != INF && visited[i] == false) {
                    queue.offer(V[i]);
                    visited[i] = true;
                }
            }
        }
        System.out.print('\n');
    }

    public void topology() {
        int[] in_degree = new int[V.length];
        for (int i = 0; i < E.length; i++) {
            for (int j = 0; j < E[0].length; j++) {
                if (E[i][j] != 0) in_degree[j]++;
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < in_degree.length; i++) {
            if (in_degree[i] == 0) queue.offer(i);
        }
        while (!queue.isEmpty()) {
            int start = queue.poll();
            System.out.print(V[start]);
            for (int i = 0; i < in_degree.length; i++) {
                if (E[start][i] != INF) {
                    in_degree[i]--;
                    if (in_degree[i] == 0) {
                        queue.offer(i);
                    }
                }
            }
        }
    }

    /**
     * O(V2)适用于权值为非负的图的单源最短路径
     *
     * @param c
     */
    public void dijkstra(char c) {
        boolean[] flag = new boolean[V.length]; // 记录每个点有没有被访问
        int[] dist = new int[V.length]; // 记录c点到其他点的距离
        for (int i = 0; i < V.length; i++) {
            flag[i] = false;
            dist[i] = E[getPosition(c)][i]; // c点的原始相邻点
        }

        // 对顶点c初始化
        flag[getPosition(c)] = true;
        dist[getPosition(c)] = 0;

        int k = 0;
        // 思路：每次找到一个最短路径且未被访问的点，根据这个值更新目标点到其他点的路径
        for (int i = 0; i < V.length; i++) {
            int min = INF;
            // 找到最短边
            for (int j = 0; j < V.length; j++) {
                if (flag[j] == false && dist[j] < min) {
                    min = dist[j];
                    k = j;
                }
            }
            // 将点k变成可达
            flag[k] = true;
            // 将k连接的还没有被访问的点根据现有信息更新
            for (int j = 0; j < V.length; j++) {
                if (E[k][j] != INF && flag[j] == false && min + E[k][j] < dist[j]) {
                    dist[j] = min + E[k][j];
                }
            }
        }
        for (int i = 0; i < V.length; i++) {
            System.out.printf("shortest(%c, %c) = %d\n", V[getPosition(c)], V[i], dist[i]);
        }
    }

    /**
     * 算法的时间复杂度为O(E3)，空间复杂度为O(E2)。
     */
    public void floyd() {
        int[][] path = new int[E.length][E[0].length];
        int[][] dist = new int[E.length][E[0].length];

        for (int i = 0; i < E.length; i++) {
            for (int j = 0; j < E[0].length; j++) {
                dist[i][j] = E[i][j];
                path[i][j] = j;
            }
        }

        // 计算最短路径
        for (int k = 0; k < E.length; k++) {
            for (int i = 0; i < E.length; i++) {
                for (int j = 0; j < E.length; j++) {
                    // 如果经过下标为k顶点路径比原两点间路径更短，则更新dist[i][j]和path[i][j]
                    int tmp = (dist[i][k] == INF || dist[k][j] == INF) ? INF : (dist[i][k] + dist[k][j]);
                    if (dist[i][j] > tmp) {
                        // "i到j最短路径"对应的值设，为更小的一个(即经过k)
                        dist[i][j] = tmp;
                        // "i到j最短路径"对应的路径，经过k
                        path[i][j] = path[i][k];
                    }
                }
            }
        }

        // 打印floyd最短路径的结果
        System.out.printf("floyd: \n");
        for (int i = 0; i < E.length; i++) {
            for (int j = 0; j < E.length; j++)
                System.out.printf("%2d  ", dist[i][j]);
            System.out.printf("\n");
        }
    }

    /**
     * 适用于权值有负值的图的单源最短路径，并且能够检测负圈，复杂度O(VE)
     *
     * @param c
     */
    public void Bellman_Ford(char c) {

        int[] dist = new int[V.length]; //dist[i] 为i点到c的最短距离
        //初始化，将dist[c]设为0，其他的都设为无限大
        for (int i = 0; i < dist.length; i++) {
            dist[i] = INF;
        }
        dist[getPosition(c)] = 0;


        //松弛操作，遍历每条边nodenum次
        // 实际上就是暴力检测有没有更短的路径
        // j和k进行的是松弛操作，i负责计数
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < E.length; j++) {
                for (int k = 0; k < E.length; k++) {
                    if (E[j][k] != INF && dist[k] > dist[j] + E[j][k]) {
                        dist[k] = dist[j] + E[j][k];
                    }
                }
            }
        }
        // 判断存在负权回路
        // 原理：如果没有负权回路，在上面的操作之后必会收敛，如果没有收敛，说明存在负权回路
        boolean judge = false;
        for (int j = 0; j < E.length; j++) {
            for (int k = 0; k < E.length; k++) {
                if (E[j][k] != INF) {
                    if (dist[k] > dist[j] + E[j][k]) {
                        judge = true;
                    }
                }
            }
        }
        if (judge) System.out.println("存在负权回路！");
        for (int i = 0; i < V.length; i++) {
            System.out.printf("shortest(%c, %c) = %d\n", V[getPosition(c)], V[i], dist[i]);
        }
    }

    /*
    适用于权值有负值，且没有负圈的图的单源最短路径，论文中的复杂度O(kE)，
    k为每个节点进入Queue的次数，且k一般<=2，但此处的复杂度证明是有问题的，其实SPFA的最坏情况应该是O(VE).
     */
    /*
    SPFA(Shortest Path Faster Algorithm) [图的存储方式为邻接表]
    是Bellman-Ford算法的一种队列实现，减少了不必要的冗余计算。

    算法大致流程是用一个队列来进行维护。 初始时将源加入队列。 每次从队列中取出一个元素，
    并对所有与他相邻的点进行松弛，若某个相邻的点松弛成功，则将其入队。 直到队列为空时算法结束。

    它可以在O(kE)的时间复杂度内求出源点到其他所有点的最短路径，可以处理负边。
    其中k为所有顶点进队的平均次数，可以证明k一般小于等于2。

    SPFA 在形式上和BFS非常类似，不同的是BFS中一个点出了队列就不可能重新进入队列，但是SPFA中
    一个点可能在出队列之后再次被放入队列，也就是一个点改进过其它的点之后，过了一段时间可能本
    身被改进，于是再次用来改进其它的点，这样反复迭代下去。

    判断有无负环：如果某个点进入队列的次数超过V次则存在负环（SPFA无法处理带负环的图）。

    SPFA算法有两个优化算法 SLF 和 LLL：
    SLF：Small Label First 策略，设要加入的节点是j，队首元素为i，若dist(j)<dist(i)，则将j插入队首，否则插入队尾。
    LLL：Large Label Last 策略，设队首元素为i，队列中所有dist值的平均值为x，若dist(i)>x则将i插入到队尾，查找下一元素，
    直到找到某一i使得dist(i)<=x，则将i出对进行松弛操作。

    引用网上资料，SLF 可使速度提高 15 ~ 20%；SLF + LLL 可提高约 50%。
    在实际的应用中SPFA的算法时间效率不是很稳定，为了避免最坏情况的出现，通常使用效率更加稳定的Dijkstra算法。
    */
    public void spfa(char c) {
        int[] dist = new int[V.length]; //dist[i] 为i点到c的最短距离
        //初始化，将dist[c]设为0，其他的都设为无限大
        for (int i = 0; i < dist.length; i++) {
            dist[i] = INF;
        }
        dist[getPosition(c)] = 0;

        LinkedList<Integer> queue = new LinkedList<>();
        queue.offer(getPosition(c));

        while (!queue.isEmpty()) {
            int curr = queue.poll();
            for (int i = 0; i < V.length; i++) {
                //取出队列的第一个元素，记为curr，对于每个点i，如果curr到每个点的边存在，判断能否进行松弛操作
                if (E[curr][i] != INF && dist[i] > dist[curr] + E[curr][i]) {
                    dist[i] = dist[curr] + E[curr][i];
                    if (!queue.contains(i))
                        queue.add(i);
                }
            }
        }

        for (int i = 0; i < V.length; i++) {
            System.out.printf("shortest(%c, %c) = %d\n", V[getPosition(c)], V[i], dist[i]);
        }
    }

    /*
    以c为顶点生成最小生成树，返回生成树的大小
    O(E2)
     */
    public void Prim(char c) {

        boolean[] flag = new boolean[V.length]; // 记录每个点有没有被访问
        int[] dist = new int[V.length]; // 记录所有点到当前的生成树的距离
        for (int i = 0; i < V.length; i++) {
            flag[i] = false;
            dist[i] = E[getPosition(c)][i]; // c点的原始相邻点
        }

        // 对顶点c初始化
        flag[getPosition(c)] = true;
        dist[getPosition(c)] = 0;

        // 选择n-1个顶点
        for (int i = 1; i < V.length; i++) {
            int index = 0;
            int min = INF;
            for (int j = 0; j < V.length; j++) {
                if (flag[j] == false && dist[j] < min) {
                    min = dist[j];
                    index = j;
                }
            }
            flag[index] = true;
            //执行更新，如果点距离当前点的距离更近，就更新dist
            for (int j = 0; j < V.length; j++) {
                if (flag[j] == false && dist[j] > E[index][j]) {
                    dist[j] = E[index][j];
                }
            }
        }

    }


    public static void main(String[] args) {
        char[] vexs = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
        int matrix[][] = {
                /*A*//*B*//*C*//*D*//*E*//*F*//*G*/
                /*A*/ {0, 12, INF, INF, INF, 16, 14},
                /*B*/ {12, 0, 10, INF, INF, 7, INF},
                /*C*/ {INF, 10, 0, 3, 5, 6, INF},
                /*D*/ {INF, INF, 3, 0, 4, INF, INF},
                /*E*/ {INF, INF, 5, 4, 0, 2, 8},
                /*F*/ {16, 7, 6, INF, 2, 0, 9},
                /*G*/ {14, INF, INF, INF, 8, 9, 0}};

        DirectedMap dm = new DirectedMap(vexs, matrix);
//        dm.print();
//        dm.DFS('A');
//        dm.BFS('A');
//        dm.topology();
        dm.dijkstra('A');
//        dm.floyd();
//        dm.Bellman_Ford('A');
        dm.spfa('A');
    }
}