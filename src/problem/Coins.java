package problem;

public class Coins {
    /*
    给定无数1,2,5元硬币，求总和为n的组合个数
     */
    public int AllCoinsComb(int n) {
        if (n <= 0)     return 0;
        int[] coins = {1, 2, 5};
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= n; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        int a;
        return dp[n];
//        int[][] dp = new int[coins.length][n + 1];
//        for (int i = 0; i < coins.length; i++) {
//            dp[i][1] = 1;
//        }
//        for (int i = 1; i < n + 1; i++) {
//            dp[1][i] = 1;
//        }
//        for (int i = 1; i < coins.length; i++) {
//            for (int j = 2; j < n + 1; j++) {
//                int sum = j;
//                while (sum - coins[i] >= 0) {
//                    dp[i][j] += dp[i - 1][sum];
//                    sum -= coins[i];
//                }
//            }
//        }
//        return dp[coins.length - 1][n];
    }

    public static void main(String[] args) {
        System.out.println(new Coins().AllCoinsComb(6));
    }
}
