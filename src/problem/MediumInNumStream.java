package problem;

import java.util.PriorityQueue;

public class MediumInNumStream {
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
