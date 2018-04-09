package problem;

public class LongestPalindrome {
    public void palindrome(String s) {
        char[] ch = (" " + s + " ").toCharArray();
        int n = ch.length;
        int p[][] = new int[n][n];  // 记录i左边和j右边的字符串部分的回文长度
        int maxLen = 0;

        for (int i = 1; i < n - 1; i++) {
            for (int j = n - 2; j >= i; j--) {
                if      (ch[i] == ch[j] && i < j)       p[i][j] = p[i - 1][j + 1] + 2;
                else if (ch[i] == ch[j] && i == j)      p[i][j] = p[i - 1][j + 1] + 1;
                else if (p[i - 1][j] >= p[i][j + 1])    p[i][j] = p[i - 1][j];
                else                                    p[i][j] = p[i][j + 1];
                if      (p[i][j] > maxLen)              maxLen = p[i][j];
            }
        }
        System.out.println(maxLen);
    }

    public static void main(String[] args) {
        String s = "12343241";
        new LongestPalindrome().palindrome(s);
    }
}
