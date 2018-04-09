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
                if (E[i][j] != Integer.MAX_VALUE)   EdgeNum++;
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
            if (in_degree[i] == 0)  queue.offer(i);
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

    public void dijkstra(char c) {
        boolean[] flag = new boolean[V.length];
        int[] prev = new int[V.length];
        int[] dist = new int[V.length]; // 记录c点到其他点的距离
        for (int i = 0; i < V.length; i++) {
            flag[i] = false;
            prev[i] = 0;
            dist[i] = E[getPosition(c)][i];
        }

        // 对顶点c初始化
        flag[getPosition(c)] = true;
        dist[getPosition(c)] = 0;

        int k = 0;
        for (int i = 0; i < V.length; i++) {
            int min = INF;
            for (int j = 0; j < V.length; j++) {
                if (flag[j] == false && dist[j] < min) {
                    min = dist[j];
                    k = j;
                }
            }
            flag[k] = true;
            for (int j = 0; j < V.length; j++) {
                int tmp = E[k][j] == INF ? INF : (min + E[k][j]);
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

    public static void main(String[] args) {
        char[] vexs = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
        int matrix[][] = {
                /*A*//*B*//*C*//*D*//*E*//*F*//*G*/
                /*A*/ {   0,  12, INF, INF, INF,  16,  14},
                /*B*/ {  12,   0,  10, INF, INF,   7, INF},
                /*C*/ { INF,  10,   0,   3,   5,   6, INF},
                /*D*/ { INF, INF,   3,   0,   4, INF, INF},
                /*E*/ { INF, INF,   5,   4,   0,   2,   8},
                /*F*/ {  16,   7,   6, INF,   2,   0,   9},
                /*G*/ {  14, INF, INF, INF,   8,   9,   0}};
        DirectedMap dm = new DirectedMap(vexs, matrix);
//        dm.print();
//        dm.DFS('A');
//        dm.BFS('A');
//        dm.topology();
        dm.dijkstra('A');
    }
}