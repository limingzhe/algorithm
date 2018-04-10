package problem;

public class LongestPalindrome {
    /*
    给定一个字符串s，你可以从中删除一些字符，使得剩下的串是一个回文串。如何删除才能使得回文串最长呢？
    输出需要删除的字符个数。
    */
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
