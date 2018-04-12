package problem;

import java.util.Stack;

public class EnglishFluent {
    public int simple (String s) {
        if (s.charAt(0) == '-') {
            s = '0' + s;
        }
        int num = 0;
        Stack<Integer> digitStack = new Stack<>();
        Stack<Character> opStack = new Stack<>();
        char[] c = s.toCharArray();
        for (int i = 0; i < c.length; i++) {
            if (Character.isDigit(c[i])) {
                num = num * 10 + c[i] - '0';
            } else {
                digitStack.push(num);
                num = 0;
                opStack.push(c[i]);
            }
        }
        digitStack.push(num);
        while (!opStack.isEmpty()) {
            int a = digitStack.pop();
            int b = digitStack.pop();
            char op = opStack.pop();
            if (opStack.isEmpty()) {
                if (op == '+')  return a + b;
                else return b - a;
            } else if (opStack.peek() == '-') {
                if (op == '+')  digitStack.push(b - a);
                else digitStack.push(b + a);
            } else {
                if (op == '+')  digitStack.push(b + a);
                else digitStack.push(b - a);
            }
        }
        return 0;
    }
}
