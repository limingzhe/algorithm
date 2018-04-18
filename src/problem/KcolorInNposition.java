package problem;

import java.util.Scanner;

public class KcolorInNposition {
    /*
    把k种颜色放进n个位置，且每种颜色至少占领一个位置
     */
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int N = scan.nextInt();
        int k = scan.nextInt();
        int[][] dp = new int[N + 1][k + 1];
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (j == 1) dp[i][j] = 1;
                else if (i < j) dp[i][j] = 0;
                else if (i == j) dp[i][j] = dp[i - 1][j - 1] * i % 772235;
                else dp[i][j] = j * (dp[i - 1][j] + dp[i - 1][j - 1]) % 772235;
            }
        }
        System.out.println(dp[N][k]);
    }
}
