package problem;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class StringPermutation {
    public List<String> Permutation(String str) {
        ArrayList<String> result = new ArrayList<>();
        if (str == null || str.length() == 0)
            return result;
        char[] chars = str.toCharArray();
        TreeSet<String> temp = new TreeSet<>();
        Permutation(chars, 0, temp);
        result.addAll(temp);
        return result;
    }

    public void Permutation(char[] chars, int index, TreeSet<String> result) {
        if (chars == null || chars.length == 0)
            return;
        if (index < 0 || index > chars.length - 1)
            return;
        if (index == chars.length - 1) {
            result.add(String.valueOf(chars));
            System.out.println(String.valueOf(chars));
        } else {
            for (int i = index; i <= chars.length - 1; i++) {
                swap(chars, index, i);
                Permutation(chars, index + 1, result);
                // 回退
                swap(chars, index, i);
            }
        }
    }

    public void swap(char[] c, int a, int b) {
        char temp = c[a];
        c[a] = c[b];
        c[b] = temp;
    }

    public static void main(String[] args) {
        new StringPermutation().Permutation("abc");
    }
}
