package problem;

public class LongestPalindrome {
    /*
    最长回文子串，不一定连续
    */
    public void longestPalindrome(String s) {
        int[][] dp = new int[s.length()][s.length()];
        for (int i = 0; i < s.length(); i++) {
            dp[i][i] = 1;
            for (int j = i - 1; j >= 0; j--) {
                if (s.charAt(i) == s.charAt(j)) dp[i][j] = dp[i - 1][j + 1] + 2;
                else dp[i][j] = Math.max(dp[i][j + 1], dp[i - 1][j]);
            }
        }
        System.out.println(dp[s.length() - 1][0]);
    }

    /*
    回文子序列个数
     */
    public void NumOfPalindromeSubSequence(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];

        for (int j = 0; j < len; j++) {
            dp[j][j] = 1;
            for (int i = j - 1; i >= 0; i--) {
                if (s.charAt(i) != s.charAt(j))
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
                else
                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1;
            }
        }
        System.out.println(dp[0][len - 1]);
    }

    public static void main(String[] args) {
        String s = "12343241";
        new LongestPalindrome().longestPalindrome(s);
        String s1 = "XXY";
        new LongestPalindrome().NumOfPalindromeSubSequence(s1);
    }
}
