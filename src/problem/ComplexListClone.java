package problem;

public class ComplexListClone {
    private class RandomListNode {
        int label;
        RandomListNode next;
        RandomListNode random;
        RandomListNode (int label) {
            this.label = label;
        }
    }

    public RandomListNode Clone2(RandomListNode pHead) {
        if(pHead == null)
            return null;
        RandomListNode head = new RandomListNode(pHead.label) ;
        RandomListNode temp = head ;

        while(pHead.next != null) {
            temp.next = new RandomListNode(pHead.next.label) ;
            if(pHead.random != null) {
                temp.random = new RandomListNode(pHead.random.label) ;
            }
            pHead = pHead.next ;
            temp = temp.next ;
        }
        return head ;
    }
}
