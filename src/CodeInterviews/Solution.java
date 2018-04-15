package CodeInterviews;

import java.util.*;

public class Solution {
    /*
    二维数组的查找
    在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
    请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     */
    public boolean Find(int target, int[][] array) {
        // 确定右上角的下标
        int right = array[0].length - 1;
        int top = 0;
        while (right >= 0 && top <= array.length - 1) {
            if (target == array[top][right]) {
                return true;
            } else if (target < array[top][right]) {
                right--;
            } else {
                top++;
            }
        }
        return false;
    }

    /*
    请实现一个函数，将一个字符串中的空格替换成“%20”。
    例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     */
    public String replaceSpace(StringBuffer str) {
        int numberOfSpace = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                numberOfSpace++;
            }
        }
        int newLength = str.length() + 2 * numberOfSpace;
        int oldLength = str.length();
        str.setLength(newLength);

        while (newLength != oldLength) {
            if (str.charAt(oldLength - 1) == ' ') {
                str.setCharAt(newLength - 1, '0');
                str.setCharAt(newLength - 2, '2');
                str.setCharAt(newLength - 3, '%');
                oldLength--;
                newLength -= 3;
            } else {
                str.setCharAt(newLength - 1, str.charAt(oldLength - 1));
                oldLength--;
                newLength--;
            }
        }
        return str.toString();
    }

    /*
    从尾到头打印链表
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }
        while (!stack.isEmpty()) {
            list.add(stack.pop());
        }
        return list;
    }

    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    /*
    输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
    假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
    例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0) return null;
        if (pre.length == 1) return new TreeNode(pre[0]);
        int root = pre[0];
        TreeNode node = new TreeNode(root);
        for (int i = 0; i < in.length; i++) {
            if (in[i] == root) {

                node.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, 1 + i),
                        Arrays.copyOfRange(in, 0, i));
                node.right = reConstructBinaryTree(Arrays.copyOfRange(pre, 1 + i, pre.length),
                        Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return node;
    }


    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /*
    用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
     */
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (!stack2.isEmpty()) {
            return stack2.pop();
        } else {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
            return stack2.pop();
        }
    }

    /*
    把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
    例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
    NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     */
    public int minNumberInRotateArray(int[] array) {
        int left = 0;
        int right = array.length - 1;
        int mid = 0;
        while (array[left] >= array[right]) {
            if (right - left == 1) {
                mid = right;
                break;
            }
            mid = (left + right) / 2;
            if (array[left] == array[mid] && array[right] == array[mid]) {
                int min = array[left];
                for (int i = left; i <= right; i++) {
                    if (array[i] < min) min = array[i];
                }
                return min;
            } else if (array[mid] >= array[left]) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return array[mid];
    }

    /*
    大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。
     */
    public int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;
        int a = 1;
        int b = 1;
        for (int i = 0; i < n - 2; i++) {
            int tmp = a;
            a = b;
            b = tmp + b;
        }
        return b;
    }

    /*
    一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     */
    public int JumpFloor(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        int[] dp = new int[target + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= target; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[target];
    }

    /*
    一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     */
    public int JumpFloorII(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        int[] dp = new int[target + 1];
        dp[1] = 1;
        for (int i = 2; i <= target; i++) {
            dp[i] = 2 * dp[i - 1];
        }
        return dp[target];
    }

    /*
    我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，
    总共有多少种方法？
     */
    public int RectCover(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        int[] dp = new int[target + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= target; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[target];
    }

    /*
    输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     */
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            if ((n & 1) == 1) {
                count++;
            }
            n >>>= 1;
        }
        return count;

//        链接：https://www.nowcoder.com/questionTerminal/8ee967e43c2c4ec193b040ea7fbb10b8
//        来源：牛客网
//        最优解
//
//        int count = 0;
//        while (n != 0) {
//            ++count;
//            n = (n - 1) & n;
//        }
//        return count;
    }

    /*
    给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) return 1;
        boolean isNeg = false;
        if (exponent < 0) {
            exponent = -exponent;
            isNeg = true;
        }
        double result = PowerRecursive(base, exponent);
        if (!isNeg) return result;
        else return 1 / result;
    }

    public double PowerRecursive(double base, int exponent) {
        if (exponent == 0) return 1;
        if (exponent == 1) return base;
        double result = PowerRecursive(base, exponent >> 1);
        result *= result;
        if ((exponent & 1) == 1) result *= base;
        return result;
    }

    /*
    输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
    使得所有的奇数位于数组的前半部分，
    所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     */
    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) return;
        for (int i = 0; i < array.length; i++) {  // 冒泡排序
            for (int j = 0; j < array.length - i - 1; j++) {
                if ((array[j] & 1) == 0 && (array[j + 1] & 1) == 1) {
                    int tmp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = tmp;
                }
            }
        }
    }

    public ListNode FindKthToTail(ListNode head, int k) {
        ListNode fast = head;
        ListNode slow = head;
        while (k-- > 0) {
            if (fast == null) return null;
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /*
    输入一个链表，反转链表后，输出链表的所有元素。
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null) return null;
        ListNode pre = null;
        ListNode curr = head;
        ListNode last = head.next;
        while (last != null) {
            curr.next = pre;
            pre = curr;
            curr = last;
            last = last.next;
        }
        curr.next = pre;
        return curr;
    }

    /*
    输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) return null;
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        if (list1.val >= list2.val) {
            list2.next = Merge(list1, list2.next);
            return list2;
        } else {
            list1.next = Merge(list1.next, list2);
            return list1;
        }
    }

    /*
    输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
        if (root2 != null && root1 != null) {
            //如果找到了对应Tree2的根节点的点
            if (root1.val == root2.val) {
                //以这个根节点为为起点判断是否包含Tree2
                result = doesTree1HaveTree2(root1, root2);
            }
            //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.left, root2);
            }
            //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.right, root2);
            }
        }
        //返回结果
        return result;
    }

    private boolean doesTree1HaveTree2(TreeNode node1, TreeNode node2) {
        //如果Tree2已经遍历完了都能对应的上，返回true，要先判断tree2有没遍历完成
        if (node2 == null) {
            return true;
        }
        //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
        if (node1 == null) {
            return false;
        }
        //如果其中有一个点没有对应上，返回false
        if (node1.val != node2.val) {
            return false;
        }

        //如果根节点对应的上，那么就分别去子节点里面匹配
        return doesTree1HaveTree2(node1.left, node2.left) && doesTree1HaveTree2(node1.right, node2.right);
    }

    /*
    操作给定的二叉树，将其变换为源二叉树的镜像。
    输入描述:
    二叉树的镜像定义：源二叉树
                8
               /  \
              6   10
             / \  / \
            5  7 9 11
            镜像二叉树
                8
               /  \
              10   6
             / \  / \
            11 9 7  5
     */
    public void Mirror(TreeNode root) {
        if (root == null) return;
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        Mirror(root.left);
        Mirror(root.right);
    }

    /*
    输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
    例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList list = new ArrayList();
        int rows = matrix.length;
        int columns = matrix[0].length;
        if (rows == 0 || columns == 0) return list;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            // left to right
            for (int i = left; i <= right; i++) {
                list.add(matrix[top][i]);
            }
            // right to bottom
            for (int i = top + 1; i <= bottom; i++) {
                list.add(matrix[i][right]);
            }
            // right to left
            if (top != bottom)
                for (int i = right - 1; i >= left; i--) {
                    list.add(matrix[bottom][i]);
                }
            // bottom to top
            if (left != right)
                for (int i = bottom - 1; i >= top + 1; i--) {
                    list.add(matrix[i][left]);
                }
            bottom--;
            top++;
            left++;
            right--;
        }
        return list;
    }

    /*
    定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
     */
    class minStack {
        Stack<Integer> data, min = new Stack<>();

        public void push(int node) {
            data.push(node);
            if (min.isEmpty() || node < min.peek()) {
                min.push(node);
            } else {
                min.push(min.peek());
            }

        }

        public void pop() {
            min.pop();
            data.pop();
        }

        public int top() {
            return data.peek();
        }

        public int min() {
            return min.peek();
        }
    }

    /*
    输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
    假设压入栈的所有数字均不相等。
    例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，
    但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) return false;
        Stack<Integer> stack = new Stack<>();
        // 标识弹出序列的位置
        int popIndex = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (!stack.isEmpty() && stack.peek() == popA[popIndex]) {
                stack.pop();
                popIndex++;
            }
        }
        return stack.isEmpty();
    }

    /*
    从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        if (root == null) return list;
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        return list;
    }

    /*
    输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
    如果是则输出Yes,否则输出No。
    假设输入的数组的任意两个数字都互不相同。
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0) return false;
        if (sequence.length == 1) return true;
        return judge(sequence, 0, sequence.length - 1);
    }

    private boolean judge(int[] a, int start, int end) {
        if (start >= end) return true;
        int i = start;
        while (a[i] < a[end]) i++;
        for (int j = i; j < end; j++) {
            if (a[j] < a[end]) return false;
        }
        return judge(a, start, i - 1) && judge(a, i, end - 1);
    }

    /*
    输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
    路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
     */
    private ArrayList<ArrayList<Integer>> listAll = new ArrayList<>();
    private ArrayList<Integer> list = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) return listAll;
        list.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null)
            listAll.add(new ArrayList<>(list));
        FindPath(root.left, target);
        FindPath(root.right, target);
        list.remove(list.size() - 1);
        return listAll;
    }

    private class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    /*
    输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
    返回结果为复制后复杂链表的head。
    （注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
     */
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) return null;
        RandomListNode pCur = pHead;
        //复制next 如原来是A->B->C 变成A->A'->B->B'->C->C'
        while (pCur != null) {
            RandomListNode node = new RandomListNode(pCur.label);
            node.next = pCur.next;
            pCur.next = node;
            pCur = node.next;
        }
        pCur = pHead;
        //复制random。 pCur是原来链表的结点 pCur.next是复制pCur的结点
        while (pCur != null) {
            if (pCur.random != null) {
                pCur.next.random = pCur.random.next;
            }
            pCur = pCur.next.next;
        }
        RandomListNode head = pHead.next;
        RandomListNode cur = head;
        pCur = pHead;
        //拆分链表
        while (pCur != null) {
            pCur.next = pCur.next.next;
            if (cur.next != null) {
                cur.next = cur.next.next;
            }
            cur = cur.next;
            pCur = pCur.next;
        }
        return head;
    }

    /*
    输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
     */
    public TreeNode Convert(TreeNode root) {

        if (root == null)
            return null;
        if (root.left == null && root.right == null)
            return root;
        // 1.将左子树构造成双链表，并返回链表头节点
        TreeNode left = Convert(root.left);
        TreeNode p = left;
        // 2.定位至左子树双链表最后一个节点
        while (p != null && p.right != null) {
            p = p.right;
        }
        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
        if (left != null) {
            p.right = root;
            root.left = p;
        }
        // 4.将右子树构造成双链表，并返回链表头节点
        TreeNode right = Convert(root.right);
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if (right != null) {
            right.left = root;
            root.right = right;
        }
        // 6.根据左子树链表是否为空确定返回的节点。
        return left != null ? left : root;
    }

    /*
    输入一个字符串,按字典序打印出该字符串中字符的所有排列。
    例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     */
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str != null && str.length() > 0) {
            PermutationHelper(str.toCharArray(), 0, res);
            Collections.sort(res);
        }
        return res;

    }

    public void PermutationHelper(char[] cs, int i, List<String> list) {
        if (i == cs.length - 1) {
            String val = String.valueOf(cs);
            if (!list.contains(val)) list.add(val);
        } else {
            for (int j = 0; j < cs.length; j++) {
                swap(cs, i, j);
                PermutationHelper(cs, i + 1, list);
                swap(cs, i, j);
            }
        }
    }

    public void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }

    /*
    数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
    由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。
    如果不存在则输出0。
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        int maxValue = array[0];
        int num = 1;
        for (int i = 1; i < array.length; i++) {
            if (num == 0) {
                maxValue = array[i];
                num = 1;
            } else if (maxValue == array[i]) {
                num++;
            } else {
                num--;
            }
        }
        num = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] == maxValue) {
                num++;
            }
        }
        if (num >= array.length / 2 + 1) {
            return maxValue;
        } else {
            return 0;
        }
    }

    /*
    输入n个整数，找出其中最小的K个数。
    例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (k > input.length || k == 0) return result;
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(k, Comparator.reverseOrder());
        for (int i = 0; i < input.length; i++) {
            if (maxHeap.size() != k) {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                maxHeap.poll();
                maxHeap.offer(input[i]);
            }
        }
        for (int i : maxHeap) {
            result.add(i);
        }
        return result;
    }

    /*
    连续子数组最大和
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 0) return 0;
        int max = array[0];
        int cur = array[0];
        for (int i = 1; i < array.length; i++) {
            if (cur >= 0) cur += array[i];
            else cur = array[i];
            if (cur > max) max = cur;
        }
        return max;
    }

    /*
    1~n整数中1出现的次数
     */
    public int NumberOf1Between1AndN_Solution(int n) {

        /*
        设N = abcde ,其中abcde分别为十进制中各位上的数字。
        如果要计算百位上1出现的次数，它要受到3方面的影响：百位上的数字，百位以下（低位）的数字，百位以上（高位）的数字。
        ① 如果百位上数字为0，百位上可能出现1的次数由更高位决定。比如：12013，则可以知道百位出现1的情况可能是：100~199，1100~1199,2100~2199，，...，11100~11199，一共1200个。可以看出是由更高位数字（12）决定，并且等于更高位数字（12）乘以 当前位数（100）。
        ② 如果百位上数字为1，百位上可能出现1的次数不仅受更高位影响还受低位影响。比如：12113，则可以知道百位受高位影响出现的情况是：100~199，1100~1199,2100~2199，，....，11100~11199，一共1200个。和上面情况一样，并且等于更高位数字（12）乘以 当前位数（100）。但同时它还受低位影响，百位出现1的情况是：12100~12113,一共114个，等于低位数字（113）+1。
        ③ 如果百位上数字大于1（2~9），则百位上出现1的情况仅由更高位决定，比如12213，则百位出现1的情况是：100~199,1100~1199，2100~2199，...，11100~11199,12100~12199,一共有1300个，并且等于更高位数字+1（12+1）乘以当前位数（100）。
        */
        int count = 0;//1的个数
        int i = 1;//当前位
        int current, after, before;
        while ((n / i) != 0) {
            current = (n / i) % 10; //高位数字
            before = n / (i * 10); //当前位数字
            after = n - (n / i) * i; //低位数字
            //如果为0,出现1的次数由高位决定,等于高位数字 * 当前位数
            if (current == 0)
                count += before * i;
                //如果为1,出现1的次数由高位和低位决定,高位*当前位+低位+1
            else if (current == 1)
                count += before * i + after + 1;
                //如果大于1,出现1的次数由高位决定,//（高位数字+1）* 当前位数
            else {
                count += (before + 1) * i;
            }
            //前移一位
            i = i * 10;
        }
        return count;
    }

    /*
    输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
    例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     */
    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0) return "";
        int len = numbers.length;
        String[] str = new String[len];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            str[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(str, (s1, s2) -> {
            String c1 = s1 + s2;
            String c2 = s2 + s1;
            return c1.compareTo(c2);
        });
        for (int i = 0; i < len; i++) {
            sb.append(str[i]);
        }
        return sb.toString();
    }

    /*
    把只包含因子2、3和5的数称作丑数（Ugly Number）。
    例如6、8都是丑数，但14不是，因为它包含因子7。
    习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
     */
    public int GetUglyNumber_Solution(int index) {
        if (index < 7) return index;
        int[] res = new int[index];
        res[0] = 1;
        int t2 = 0, t3 = 0, t5 = 0;
        for (int i = 1; i < index; i++) {
            res[i] = Math.min(res[t2] * 2, Math.min(res[t3] * 3, res[t5] * 5));
            if (res[i] == res[t2] * 2) t2++;
            if (res[i] == res[t3] * 3) t3++;
            if (res[i] == res[t5] * 5) t5++;
        }
        return res[index - 1];
    }

    /*
    在一个字符串(1<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置
     */
    public int FirstNotRepeatingChar(String str) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (map.containsKey(c)) {
                map.put(c, map.get(c) + 1);
            } else {
                map.put(c, 1);
            }
        }
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (map.get(c) == 1) {
                return i;
            }
        }
        return -1;
    }

    /*
    在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
    并将P对1000000007取模的结果输出。 即输出P%1000000007
     */
    public int InversePairs(int[] array) {
        if (array == null || array.length <= 0) {
            return 0;
        }
        int pairsNum = mergeSort(array, 0, array.length - 1);
        return pairsNum;
    }


    private int mergeSort(int[] array, int left, int right) {
        if (left == right) {
            return 0;
        }
        int mid = (left + right) / 2;
        int leftNum = mergeSort(array, left, mid);
        int rightNum = mergeSort(array, mid + 1, right);
        return (Sort(array, left, mid, right) + leftNum + rightNum) % 1000000007;
    }


    private int Sort(int[] a, int start, int mid, int end) {
        int cnt = 0;
        int[] tmp = new int[end - start + 1];

        int i = start, j = mid + 1, k = 0;
        while (i <= mid && j <= end) {
            if (a[i] <= a[j])
                tmp[k++] = a[i++];
            else {
                tmp[k++] = a[j++];
                cnt += mid - i + 1;// core code, calculate InversePairs............
                if (cnt > 1000000007) cnt %= 1000000007;
            }
        }

        while (i <= mid)
            tmp[k++] = a[i++];
        while (j <= end)
            tmp[k++] = a[j++];
        for (k = 0; k < tmp.length; k++)
            a[start + k] = tmp[k];
        return cnt;
    }

    /*
    输入两个链表，找出它们的第一个公共结点。
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = (p1 == null ? pHead2 : p1.next);
            p2 = (p2 == null ? pHead1 : p2.next);
        }
        return p1;
    }

    /*
    统计一个数字在排序数组中出现的次数。
     */

    public int GetNumberOfK(int[] array, int k) {
        int length = array.length;
        if (length == 0) {
            return 0;
        }
        int firstK = getFirstK(array, k, 0, length - 1);
        int lastK = getLastK(array, k, 0, length - 1);
        if (firstK != -1 && lastK != -1) {
            return lastK - firstK + 1;
        }
        return 0;
    }

    //递归写法
    private int getFirstK(int[] array, int k, int start, int end) {
        if (start > end) {
            return -1;
        }
        int mid = (start + end) >> 1;
        if (array[mid] > k) {
            return getFirstK(array, k, start, mid - 1);
        } else if (array[mid] < k) {
            return getFirstK(array, k, mid + 1, end);
        } else if (mid - 1 >= 0 && array[mid - 1] == k) {
            return getFirstK(array, k, start, mid - 1);
        } else {
            return mid;
        }
    }

    //循环写法
    private int getLastK(int[] array, int k, int start, int end) {
        int length = array.length;
        int mid = (start + end) >> 1;
        while (start <= end) {
            if (array[mid] > k) {
                end = mid - 1;
            } else if (array[mid] < k) {
                start = mid + 1;
            } else if (mid + 1 < length && array[mid + 1] == k) {
                start = mid + 1;
            } else {
                return mid;
            }
            mid = (start + end) >> 1;
        }
        return -1;
    }

    /*
    如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
    如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
     */
    class MediumInNumStream {
        private int count = 0;  // 数据流中的数据个数
        // 优先队列集合实现了堆，默认实现的小根堆
        private PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        // 定义大根堆，更改比较方式
        private PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> {
            return o2 - o1;     // o1 - o2 则是小根堆
        });

        public void Insert(Integer num) {
            if ((count & 1) == 0) {
                // 当数据总数为偶数时，新加入的元素，应当进入小根堆
                // （注意不是直接进入小根堆，而是经大根堆筛选后取大根堆中最大元素进入小根堆）
                // 1.新加入的元素先入到大根堆，由大根堆筛选出堆中最大的元素
                maxHeap.offer(num);
                int filteredMaxNum = maxHeap.poll();
                // 2.筛选后的【大根堆中的最大元素】进入小根堆
                minHeap.offer(filteredMaxNum);
            } else {
                // 当数据总数为奇数时，新加入的元素，应当进入大根堆
                // （注意不是直接进入大根堆，而是经小根堆筛选后取小根堆中最大元素进入大根堆）
                // 1.新加入的元素先入到小根堆，由小根堆筛选出堆中最小的元素
                minHeap.offer(num);
                int filteredMinNum = minHeap.poll();
                // 2.筛选后的【小根堆中的最小元素】进入小根堆
                maxHeap.offer(filteredMinNum);
            }
            count++;
        }


        public Double GetMedian() {
            // 数目为偶数时，中位数为小根堆堆顶元素与大根堆堆顶元素和的一半
            if ((count & 1) == 0) {
                return new Double((minHeap.peek() + maxHeap.peek())) / 2;
            } else {
                return new Double(minHeap.peek());
            }
        }

    }

    public int TreeDepth(TreeNode root) {
        if (root == null) return 0;
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        return Math.max(left, right) + 1;
    }

    public boolean IsBalanced_Solution(TreeNode root) {
        return getDepth(root) != -1;
    }

    private int getDepth(TreeNode root) {
        if (root == null) return 0;
        int leftHeight = getDepth(root.left);
        if (leftHeight == -1) return -1;
        int rightHeight = getDepth(root.right);
        if (rightHeight == -1) return -1;
        return Math.abs(leftHeight - rightHeight) > 1 ? -1 : 1 + Math.max(leftHeight, rightHeight);
    }

    /*
    一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
    //num1,num2分别为长度为1的数组。传出参数
    //将num1[0],num2[0]设置为返回结果
     */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        if (array == null || array.length <= 1) {
            num1[0] = num2[0] = 0;
            return;
        }
        int sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum ^= array[i];
        }
        int index = 0;
        // 找到从右往左第一个1
        for (int i = 0; i < 32; i++) {
            if ((sum & (1 << i)) != 0) {
                index = i;
                break;
            }
        }
        // 根据这第一个1把数组分为两部分
        for (int i = 0; i < array.length; i++) {
            if ((array[i] & (1 << index)) != 0) {
                num2[0] ^= array[i];
            } else {
                num1[0] ^= array[i];
            }
        }
    }


    /*
     * 数组a中只有一个数出现一次，其他数都出现了2次，找出这个数字
     *
     */
    public static int find1From2(int[] a) {
        int len = a.length, res = 0;
        for (int i = 0; i < len; i++) {
            res = res ^ a[i];
        }
        return res;
    }


    /*
     * 数组a中只有一个数出现一次，其他数字都出现了3次，找出这个数字
     */
    public static int find1From3(int[] a) {
        int[] bits = new int[32];
        int len = a.length;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < 32; j++) {
                bits[j] = bits[j] + ((a[i] >>> j) & 1);
            }
        }
        int res = 0;
        for (int i = 0; i < 32; i++) {
            if (bits[i] % 3 != 0) {
                res = res | (1 << i);
            }
        }
        return res;
    }

    /*
    输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int start = 1, end = 2;
        while (start != (sum + 1) / 2) {
            int localSum = 0;
            for (int i = start; i <= end; i++) {
                localSum += i;
            }
            if (localSum == sum) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = start; i <= end; i++) {
                    list.add(i);
                }
                res.add(list);
                start++;
                end++;
            } else if (localSum > sum) {
                start++;
            } else {
                end++;
            }
        }
        return res;
    }

    /*
    输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，
    如果有多对数字的和等于S，输出两个数的乘积最小的。
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> list = new ArrayList<>();
        if (array == null || array.length < 2) {
            return list;
        }
        int i = 0, j = array.length - 1;
        while (i < j) {
            if (array[i] + array[j] == sum) {
                list.add(array[i]);
                list.add(array[j]);
                return list;
            } else if (array[i] + array[j] > sum) {
                j--;
            } else {
                i++;
            }
        }

        return list;
    }

    /*
    对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
    例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
     */
    public String LeftRotateString(String str, int n) {
        if (str == null) return null;
        if (str.length() == 0 || n == 0) return str;
        char[] c = str.toCharArray();
        int length = c.length;
        swapArray(c, 0, length - 1);
        swapArray(c, 0, length - 1 - n);
        swapArray(c, length - n, length - 1);
        return String.valueOf(c);
    }

    private void swapArray(char[] c, int start, int end) {
        for (int i = 0; i < (end - start + 1) / 2; i++) {
            char tmp = c[start + i];
            c[start + i] = c[end - i];
            c[end - i] = tmp;
        }
    }

    /*
    输入："I am a student."

    输出："student. a am I"
     */
    public String ReverseSentence(String str) {
        if (str.length() == 0) return str;
        int n = str.length();
        char[] c = str.toCharArray();
        int i = 0, j = 0;
        while (j <= n) {
            if (j == n || c[j] == ' ') {
                swapArray(c, i, j - 1);
                i = j + 1;
            }
            j++;
        }
        swapArray(c, 0, n - 1);
        return new String(c);
    }

    /*
    五张牌，其中大小鬼为癞子，牌面大小为 0。判断是否能组成顺子。
     */
    public boolean isContinuous(int[] numbers) {
        if (numbers.length < 5) return false;
        // 排序
        Arrays.sort(numbers);
        // 统计癞子个数
        int cnt = 0;
        for (int num : numbers) {
            if (num == 0) cnt++;
        }
        for (int i = cnt; i < numbers.length - 1; i++) {
            if (numbers[i + 1] == numbers[i]) return false;
            int interval = numbers[i + 1] - numbers[i] - 1;
            if (interval > cnt) return false;
            cnt -= interval;
        }
        return true;
    }

    /*
    让小朋友们围成一个大圈。然后，他随机指定一个数 m，让编号为 0 的小朋友开始报数。
    每次喊到 m-1 的那个小朋友要出列唱首歌，然后可以在礼品箱中任意的挑选礼物，并且不再回到圈中，
    从他的下一个小朋友开始，继续 0...m-1 报数 .... 这样下去 .... 直到剩下最后一个小朋友，可以不用表演。

    约瑟夫环，圆圈长度为 n 的解可以看成长度为 n-1 的解再加上报数的长度 m。因为是圆圈，所以最后需要对 n 取余。
     */
    public int LastRemaining_Solution(int n, int m) {
        if (n == 0) return -1;
        if (n == 1) return 0;
        return (LastRemaining_Solution(n - 1, m) + m) % n;
    }

    /*
    求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     */
    public int Sum_Solution(int n) {
        int sum = n;
        boolean b = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
        return sum;
    }

    /*
    写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
     */
    public int Add(int num1, int num2) {
        if (num2 == 0) return num1;
        return Add(num1 ^ num2, (num1 & num2) << 1);
    }

    /*
    将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
     */
    public int StrToInt(String str) {
        int res = 0;
        boolean isNegtive = false;
        for (int i = 0; i < str.length(); i++) {
            if (Character.isDigit(str.charAt(i))) {
                res = res * 10 + str.charAt(i) - '0';
            } else if (i == 0 && str.charAt(i) == '+') {
                continue;
            } else if (i == 0 && str.charAt(i) == '-') {
                isNegtive = true;
            } else {
                return 0;
            }
        }
        return isNegtive ? -res : res;
    }

    /*
    在一个长度为n的数组里的所有数字都在0到n-1的范围内。
    数组中某些数字是重复的，但不知道有几个数字是重复的。
    也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
    例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
     */
    /*
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
     */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        for (int i = 0; i < length; i++) {
            if (i == numbers[i]) {
                continue;
            } else {
                while (numbers[i] != i && numbers[i] != numbers[numbers[i]]) {
                    swap(numbers, i, numbers[i]);
                }
                if (numbers[i] != i && numbers[i] == numbers[numbers[i]]) {
                    duplication[0] = numbers[i];
                    return true;
                }
            }
        }
        return false;
    }

    private void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /*
    给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
    其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。
    不能使用除法。
     */
    public int[] multiply(int[] A) {
        int n = A.length;
        int[] B = new int[n];
        for (int i = 0, product = 1; i < n; product *= A[i], i++) {
            B[i] = product;
        }
        for (int i = n - 1, product = 1; i >= 0; product *= A[i], i--) {
            B[i] *= product;
        }
        return B;
    }

    /*
    请实现一个函数用来匹配包括'.'和'*'的正则表达式。
    模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
    在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，
    但是与"aa.a"和"ab*a"均不匹配

     */
    public boolean match(char[] str, char[] pattern) {
        int m = str.length, n = pattern.length;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if (pattern[i - 1] == '*') {
                dp[0][i] = dp[0][i - 2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str[i - 1] == pattern[j - 1] || pattern[j - 1] == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (pattern[j - 1] == '*') {
                    if (pattern[j - 2] == str[i - 1] || pattern[j - 2] == '.') {
                        dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                    } else {
                        dp[i][j] = dp[i][j - 2];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /*
    请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
    例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
    但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
     */
    public boolean isNumeric(char[] str) {
        return new String(str).matches("[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?");
    }

    class FirstAppearingOnce {
        BitSet bs1 = new BitSet(256);
        BitSet bs2 = new BitSet(256);
        StringBuffer s = new StringBuffer();

        //Insert one char from stringstream
        public void Insert(char ch) {
            s.append(ch);
            if (!bs1.get(ch) && !bs2.get(ch)) {
                bs1.set(ch);
            } else if (bs1.get(ch) && !bs2.get(ch)) {
                bs2.set(ch);
            }
        }

        //return the first appearence once char in current stringstream
        public char FirstAppearingOnce() {
            char[] str = s.toString().toCharArray();
            for (char c : str) {
                if (bs1.get(c) && !bs2.get(c)) {
                    return c;
                }
            }
            return '#';
        }
    }

    /*
    一个链表中包含环，请找出该链表的环的入口结点。
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null) return null;
        ListNode slow = pHead, fast = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = pHead;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow;
            }
        }
        return null;
    }

    /*
    在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
    例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
     */
    public ListNode deleteDuplication(ListNode pHead) {
        ListNode dummy = new ListNode(-1);//设置一个trick
        dummy.next = pHead;

        ListNode p = pHead;
        ListNode pre = dummy;
        while (p != null && p.next != null) {
            if (p.val == p.next.val) {
                int val = p.val;
                while (p != null && p.val == val)
                    p = p.next;
                pre.next = p;
            } else {
                pre = p;
                p = p.next;
            }
        }
        return dummy.next;
    }

    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }

    /*
    给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
    注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        } else {
            while (pNode.next != null) {
                TreeLinkNode parent = pNode.next;
                if (parent.left == pNode) {
                    return parent;
                }
                pNode = pNode.next;
            }
        }
        return null;
    }

    /*
    请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
     */
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) return true;
        return isSymmetrical(pRoot.left, pRoot.right);
    }

    private boolean isSymmetrical(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        if (t1.val != t2.val) return false;
        return isSymmetrical(t1.left, t2.right) && isSymmetrical(t1.right, t2.left);
    }

    /*
    请实现一个函数按照之字形打印二叉树，
    即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) return ret;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        boolean reverse = false;
        while (!queue.isEmpty()) {
            int cnt = queue.size();
            ArrayList<Integer> list = new ArrayList<>();
            for (int i = 0; i < cnt; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            if (reverse) Collections.reverse(list);
            reverse = !reverse;
            ret.add(list);
        }
        return ret;
    }

    /*
    从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
     */
    ArrayList<ArrayList<Integer>> Print1(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) return ret;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        while (!queue.isEmpty()) {
            int cnt = queue.size();
            ArrayList<Integer> list = new ArrayList<>();
            for (int i = 0; i < cnt; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            ret.add(list);
        }
        return ret;
    }

    /*
    序列化二叉树
    思路：前序遍历，遇到null设为#
     */
    class Serialize {
        public int index = -1;

        public String Serialize(TreeNode root) {
            StringBuffer sb = new StringBuffer();
            if (root == null) {
                sb.append("#,");
                return sb.toString();
            }
            sb.append(root.val + ",");
            sb.append(Serialize(root.left));
            sb.append(Serialize(root.right));
            return sb.toString();
        }

        public TreeNode Deserialize(String str) {
            index++;
            int len = str.length();
            if (index >= len) {
                return null;
            }
            String[] strr = str.split(",");
            TreeNode node = null;
            if (!strr[index].equals("#")) {
                node = new TreeNode(Integer.valueOf(strr[index]));
                node.left = Deserialize(str);
                node.right = Deserialize(str);
            }

            return node;
        }


    }

    /*
        给定一颗二叉搜索树，请找出其中的第k大的结点。
        例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。
         */
    class KthNode {
        private TreeNode ret;
        private int cnt = 0;

        public TreeNode KthNode(TreeNode pRoot, int k) {
            inOrder(pRoot, k);
            return ret;
        }

        private void inOrder(TreeNode root, int k) {
            if (root == null) return;
            if (cnt > k) return;
            inOrder(root.left, k);
            cnt++;
            if (cnt == k) ret = root;
            inOrder(root.right, k);
        }
    }

    /*
    给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
    例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，
    他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个：
    {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}，
    {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (size > num.length || size < 1) return ret;

        // 构建最大堆，即堆顶元素是堆的最大值。
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        for (int i = 0; i < size; i++)
            heap.add(num[i]);

        ret.add(heap.peek());
        for (int i = 1; i + size - 1 < num.length; i++) {
            heap.remove(num[i - 1]);
            heap.add(num[i + size - 1]);
            ret.add(heap.peek());
        }
        return ret;
    }

    /*
    请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
    路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
    如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
    例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，
    但是矩阵中不包含"abcb"路径，
    因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
     */
    class hasPath {
        public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
            int flag[] = new int[matrix.length];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (helper(matrix, rows, cols, i, j, str, 0, flag))
                        return true;
                }
            }
            return false;
        }

        private boolean helper(char[] matrix, int rows, int cols, int i, int j, char[] str, int k, int[] flag) {
            int index = i * cols + j;
            if (i < 0 || i >= rows || j < 0 || j >= cols || matrix[index] != str[k] || flag[index] == 1)
                return false;
            if (k == str.length - 1) return true;
            flag[index] = 1;
            if (helper(matrix, rows, cols, i - 1, j, str, k + 1, flag)
                    || helper(matrix, rows, cols, i + 1, j, str, k + 1, flag)
                    || helper(matrix, rows, cols, i, j - 1, str, k + 1, flag)
                    || helper(matrix, rows, cols, i, j + 1, str, k + 1, flag)) {
                return true;
            }
            flag[index] = 0;
            return false;
        }
    }

    /*
    地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
    但是不能进入行坐标和列坐标的数位之和大于k的格子。
    例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
    但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
     */
    class movingCount {
        public int movingCount(int threshold, int rows, int cols) {
            int flag[][] = new int[rows][cols]; //记录是否已经走过
            return helper(0, 0, rows, cols, flag, threshold);
        }

        private int helper(int i, int j, int rows, int cols, int[][] flag, int threshold) {
            if (i < 0 || i >= rows || j < 0 || j >= cols
                    || numSum(i) + numSum(j) > threshold
                    || flag[i][j] == 1)
                return 0;
            flag[i][j] = 1;
            return helper(i - 1, j, rows, cols, flag, threshold)
                    + helper(i + 1, j, rows, cols, flag, threshold)
                    + helper(i, j - 1, rows, cols, flag, threshold)
                    + helper(i, j + 1, rows, cols, flag, threshold) + 1;
        }


        private int numSum(int i) {
            int sum = 0;
            do {
                sum += i % 10;
            } while ((i = i / 10) > 0);
            return sum;
        }
    }
}

