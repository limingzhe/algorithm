package problem;

import java.util.PriorityQueue;

class ListNode implements Comparable<ListNode> {
    int value;
    ListNode next;

    ListNode(int val) {
        value = val;
    }

    public int compareTo(ListNode other) {
        return value - other.value;
    }
}

public class MergeKsortedLists {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0)
            return null;
        if (lists.length == 1)
            return lists[0];


        PriorityQueue<ListNode> PQ = new PriorityQueue<>();

        ListNode head = new ListNode(0);
        ListNode phead = head;

        for (ListNode list : lists) {
            if (list != null)
                PQ.offer(list);
        }
        while (!PQ.isEmpty()) {
            ListNode temp = PQ.poll();
            phead.next = temp;
            phead = phead.next;

            if (temp.next != null)
                PQ.offer(temp.next);
        }
        return head.next;
    }
}
