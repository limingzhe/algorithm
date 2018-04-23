package leetcode;

import java.util.HashMap;
import java.util.Map;

public class LeetCode1To10 {
    /*
    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Example:

    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].
     */
    /*
    思路：hash表记录《值，下标》，对于当前数，如果 结果-自己 在哈希表里，说明符合条件，否则把自己加入哈希表
     */
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                res[0] = map.get(target - nums[i]);
                res[1] = i;
                return res;
            } else {
                map.put(nums[i], i);
            }
        }
        return res;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    /*
    You are given two non-empty linked lists representing two non-negative integers.
    The digits are stored in reverse order and each of their nodes contain a single digit.
    Add the two numbers and return it as a linked list.

    You may assume the two numbers do not contain any leading zero, except the number 0 itself.

    Example

    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output: 7 -> 0 -> 8
    Explanation: 342 + 465 = 807.
     */
    /*
    思路：题目里是反向存储，设置一个进位即可，符合三个条件里任何一个while循环都需要进行下去
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = dummyHead;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int sum = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;
            carry = sum / 10;
            p.next = new ListNode(sum % 10);
            p = p.next;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        return dummyHead.next;
    }

    /*
    Given a string, find the length of the longest substring without repeating characters.

    Examples:

    Given "abcabcbb", the answer is "abc", which the length is 3.

    Given "bbbbb", the answer is "b", with the length of 1.

    Given "pwwkew", the answer is "wke", with the length of 3.
    Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
     */
    /*
    最长不重复连续子串问题，使用滑动窗口，每次更新 前窗口，如果有重复的可能更新后窗口
     */
    public int lengthOfLongestSubstring(String s) {
        int ans = 0;
        Map<Character, Integer> map = new HashMap<>();
        int lo = 0;
        for (int hi = 0; hi < s.length(); hi++) {
            if (map.containsKey(s.charAt(hi))) {
                lo = Math.max(lo, map.get(s.charAt(hi)) + 1);
            }
            ans = Math.max(hi - lo + 1, ans);
            map.put(s.charAt(hi), hi);
        }
        return ans;
    }

    /*
    There are two sorted arrays nums1 and nums2 of size m and n respectively.

    Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

    Example 1:
    nums1 = [1, 3]
    nums2 = [2]

    The median is 2.0
    Example 2:
    nums1 = [1, 2]
    nums2 = [3, 4]

    The median is (2 + 3)/2 = 2.5
     */
    /*
    思路：要求O(log (m+n))复杂度
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int left = (n + m + 1) / 2;
        int right = (n + m + 2) / 2;
        return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;
    }

    private int getKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
        if (len1 == 0) return nums2[start2 + k - 1];
        if (k == 1) return Integer.min(nums1[start1], nums2[start2]);

        int i = start1 + Integer.min(len1, k / 2) - 1;
        int j = start2 + Integer.min(len2, k / 2) - 1;
        //Eliminate half of the elements from one of the smaller arrays
        if (nums1[i] > nums2[j]) {
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        } else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }

    /*
    Given a string s, find the longest palindromic substring in s.
    You may assume that the maximum length of s is 1000.

    Example 1:

    Input: "babad"
    Output: "bab"
    Note: "aba" is also a valid answer.
    Example 2:

    Input: "cbbd"
    Output: "bb"
     */
    /*
    思路：对于字符串每个元素，把它当成回文串中心，然后向两边扩展，注意要分奇偶情况
     */
    private class LongestPalindrome {
        private int lo, maxLen;

        public String longestPalindrome(String s) {
            if (s.length() < 2) return s;
            for (int i = 0; i < s.length() - 1; i++) {
                extendPalindrome(s, i, i);  //assume odd length, try to extend Palindrome as possible
                extendPalindrome(s, i, i + 1); //assume even length.
            }
            return s.substring(lo, lo + maxLen);
        }

        private void extendPalindrome(String s, int j, int k) {
            while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
                j--;
                k++;
            }
            if (maxLen < k - j - 1) {
                lo = j + 1;
                maxLen = k - j - 1;
            }
        }
    }

    /*
    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:
    (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"

    Write the code that will take a string and make this conversion given a number of rows:

    string convert(string s, int numRows);
    Example 1:

    Input: s = "PAYPALISHIRING", numRows = 3
    Output: "PAHNAPLSIIGYIR"
    Example 2:

    Input: s = "PAYPALISHIRING", numRows = 4
    Output: "PINALSIGYAHRPI"
    Explanation:

    P     I    N
    A   L S  I G
    Y A   H R
    P     I
     */
    public String convert(String s, int numRows) {
        char[] c = s.toCharArray();
        int len = c.length;
        StringBuffer[] sb = new StringBuffer[numRows];
        for (int i = 0; i < sb.length; i++) sb[i] = new StringBuffer();

        int i = 0;
        while (i < len) {
            for (int idx = 0; idx < numRows && i < len; idx++) // vertically down
                sb[idx].append(c[i++]);
            for (int idx = numRows - 2; idx >= 1 && i < len; idx--) // obliquely up
                sb[idx].append(c[i++]);
        }
        for (int idx = 1; idx < sb.length; idx++)
            sb[0].append(sb[idx]);
        return sb[0].toString();
    }

    /*
    Given a 32-bit signed integer, reverse digits of an integer.

    Example 1:

    Input: 123
    Output: 321
    Example 2:

    Input: -123
    Output: -321
    Example 3:

    Input: 120
    Output: 21
    Note:
    Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1].
    For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
     */
    public int reverse(int x) {
        int res = 0;

        while (x != 0) {
            int tail = x % 10;
            int newRes = res * 10 + tail;
            if ((newRes - tail) / 10 != res)
                return 0;
            res = newRes;
            x /= 10;
        }
        return res;
    }

    /*
    Implement atoi which converts a string to an integer.

    The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

    The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

    If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

    If no valid conversion could be performed, a zero value is returned.

    Note:
    Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. If the numerical value is out of the range of representable values, INT_MAX (231 − 1) or INT_MIN (−231) is returned.

    Example 1:

    Input: "42"
    Output: 42
    Example 2:

    Input: "   -42"
    Output: -42
    Explanation: The first non-whitespace character is '-', which is the minus sign.
                 Then take as many numerical digits as possible, which gets 42.
    Example 3:

    Input: "4193 with words"
    Output: 4193
    Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.
    Example 4:

    Input: "words and 987"
    Output: 0
    Explanation: The first non-whitespace character is 'w', which is not a numerical
                 digit or a +/- sign. Therefore no valid conversion could be performed.
    Example 5:

    Input: "-91283472332"
    Output: -2147483648
    Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
                 Thefore INT_MIN (−231) is returned.
     */
    public int myAtoi(String str) {
        if (str == null || str.length() == 0)
            return 0;//
        str = str.trim();
        char firstChar = str.charAt(0);
        int sign = 1, start = 0, len = str.length();
        long sum = 0;
        if (firstChar == '+') {
            sign = 1;
            start++;
        } else if (firstChar == '-') {
            sign = -1;
            start++;
        }
        for (int i = start; i < len; i++) {
            if (!Character.isDigit(str.charAt(i)))
                return (int) sum * sign;
            sum = sum * 10 + str.charAt(i) - '0';
            if (sign == 1 && sum > Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            if (sign == -1 && (-1) * sum < Integer.MIN_VALUE)
                return Integer.MIN_VALUE;
        }

        return (int) sum * sign;
    }

    /*
    Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

    Example 1:

    Input: 121
    Output: true
    Example 2:

    Input: -121
    Output: false
    Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
    Example 3:

    Input: 10
    Output: false
    Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
    Follow up:

    Coud you solve it without converting the integer to a string?
     */
    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0))
            return false;

        int revertNum = 0;
        while (x > revertNum) {

            revertNum = revertNum * 10 + x % 10;
            x /= 10;
        }

        return revertNum == x || x == revertNum / 10;
    }

    /*
    Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.

    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.
    The matching should cover the entire input string (not partial).

    Note:

    s could be empty and contains only lowercase letters a-z.
    p could be empty and contains only lowercase letters a-z, and characters like . or *.
    Example 1:

    Input:
    s = "aa"
    p = "a"
    Output: false
    Explanation: "a" does not match the entire string "aa".
    Example 2:

    Input:
    s = "aa"
    p = "a*"
    Output: true
    Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
    Example 3:

    Input:
    s = "ab"
    p = ".*"
    Output: true
    Explanation: ".*" means "zero or more (*) of any character (.)".
    Example 4:

    Input:
    s = "aab"
    p = "c*a*b"
    Output: true
    Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".
    Example 5:

    Input:
    s = "mississippi"
    p = "mis*is*p*."
    Output: false
     */
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        for (int i = 0; i < p.length(); i++) {
            if (p.charAt(i) == '*' && dp[0][i - 1]) {
                dp[0][i + 1] = true;
            }
        }
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < p.length(); j++) {
                if (p.charAt(j) == '.') {
                    dp[i + 1][j + 1] = dp[i][j];
                }
                if (p.charAt(j) == s.charAt(i)) {
                    dp[i + 1][j + 1] = dp[i][j];
                }
                if (p.charAt(j) == '*') {
                    if (p.charAt(j - 1) != s.charAt(i) && p.charAt(j - 1) != '.') {
                        dp[i + 1][j + 1] = dp[i + 1][j - 1];
                    } else {
                        dp[i + 1][j + 1] = (dp[i + 1][j] || dp[i][j + 1] || dp[i + 1][j - 1]);
                    }
                }
            }
        }
        return dp[s.length()][p.length()];
    }
}
