package problem;

public class Coins {
    /*
    给定无数1,2,5元硬币，求总和为n的组合个数
     */
    public int AllCoinsComb(int n) {
        if (n <= 0)     return 0;
        int[] coins = {1, 2, 5};
//        int[] dp = new int[n + 1];
//        dp[0] = 1;
//        for (int i = 0; i < coins.length; i++) {
//            for (int j = coins[i]; j <= n; j++) {
//                dp[j] += dp[j - coins[i]];
//            }
//        }
//        return dp[n];

        // dp[i][j] = dp[i-1][j] + dp[i-1][j-coins(i)] + dp[i-1][j-2*coins(i)] + ...
        // 前i个硬币组成的j
        int[][] dp = new int[coins.length][n + 1];
        for (int i = 1; i <= n; i++) {
            dp[0][n] = n;
        }
        for (int i = 1; i < coins.length; i++) {
            for (int j = 1; j <= n; j++) {
                int tmp = j;
                dp[i][j] += dp[i-1][j];
                while (tmp - coins[i] > 0) {
                    tmp -= coins[i];
                    dp[i][j] += dp[i-1][tmp];
                }
            }
        }
        return dp[coins.length - 1][n];
    }

    public static void main(String[] args) {
        System.out.println(new Coins().AllCoinsComb(6));
    }
}
