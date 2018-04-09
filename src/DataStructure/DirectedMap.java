package DataStructure;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

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
     * O(n2)适用于权值为非负的图的单源最短路径
     * @param c
     */
    public void dijkstra(char c) {
        boolean[] flag = new boolean[V.length]; // 记录每个点有没有被访问
        int[] prev = new int[V.length];
        int[] dist = new int[V.length]; // 记录c点到其他点的距离
        for (int i = 0; i < V.length; i++) {
            flag[i] = false;
            prev[i] = 0;
            dist[i] = E[getPosition(c)][i]; // c点的原始相邻点
        }

        // 对顶点c初始化
        flag[getPosition(c)] = true;
        dist[getPosition(c)] = 0;

        int k = 0;
        for (int i = 0; i < V.length; i++) {
            int min = INF;
            // 找到最短边
            for (int j = 0; j < V.length; j++) {
                if (flag[j] == false && dist[j] < min) {
                    min = dist[j];
                    k = j;
                }
            }
            flag[k] = true;
            // 点k可达，将k连接的点变为c点的最短路径
            for (int j = 0; j < V.length; j++) {
                int tmp = (E[k][j] == INF ? INF : (min + E[k][j]));
                if (flag[j] == false && (tmp < dist[j])) {
                    dist[j] = tmp;
                    prev[j] = k;
                }
            }
        }
        for (int i = 0; i < V.length; i++) {
            System.out.printf("shortest(%c, %c) = %d\n", V[getPosition(c)], V[i], dist[i]);
        }
    }

    /**
     * 算法的时间复杂度为O(N3)，空间复杂度为O(N2)。
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
     * 可以求含负权图的单源最短路径，时间复杂度O(VE)
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
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < E.length; j++) {
                for (int k = 0; k < E.length; k++) {
                    if (E[j][k] != INF) {
                        if (dist[k] > dist[j] + E[j][k]) {
                            dist[k] = dist[j] + E[j][k];
                        }
                    }
                }
            }
        }
        // 判断存在负权回路
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
//        dm.dijkstra('A');
//        dm.floyd();
        dm.Bellman_Ford('A');
    }
}