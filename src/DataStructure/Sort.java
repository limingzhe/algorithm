package DataStructure;


/**
 * 十大排序算法
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 常见的快速排序、归并排序、堆排序、冒泡排序等属于比较排序。在排序的最终结果里，元素之间的次序依赖于它们之间的比较。每个数都必须和其他数进行比较，才能确定自己的位置。
 * 在冒泡排序之类的排序中，问题规模为n，又因为需要比较n次，所以平均时间复杂度为O(n²)。在归并排序、快速排序之类的排序中，问题规模通过分治法消减为logN次，所以时间复杂度平均O(nlogn)。
 * 比较排序的优势是，适用于各种规模的数据，也不在乎数据的分布，都能进行排序。可以说，比较排序适用于一切需要排序的情况。
 * <p>
 * 计数排序、基数排序、桶排序则属于非比较排序。非比较排序是通过确定每个元素之前，应该有多少个元素来排序。针对数组arr，计算arr[i]之前有多少个元素，则唯一确定了arr[i]在排序后数组中的位置。
 * 非比较排序只要确定每个元素之前的已有的元素个数即可，所有一次遍历即可解决。算法时间复杂度O(n)。
 * 非比较排序时间复杂度底，但由于非比较排序需要占用空间来确定唯一位置。所以对数据规模和数据分布有一定的要求。
 */
public class Sort {
    public static void main(String[] args) {
        int[] arr = {2, 8, 10, 5, 6, 3, 4, 2, 7, 5};
        int[] result = BubbleSort(arr);
//        int[] result = SelectionSort(arr);
//        int[] result = InsertionSort(arr);
//        int[] result = ShellSort(arr);
//        int[] result = MergeSort(arr);
//        int[] result = HeapSort(arr);
//        int[] result = QuickSort(arr);
//        int[] result = CountingSort(arr);
//        int[] result = BucketSort(arr);
//        int[] result = RadixSort(arr);
        for (Integer i : result) System.out.print(i + " ");
    }

    /*
    每一轮访问前i个数，相邻的互相比较，大的沉下去，最后一个数必为最大
    稳定
     */
    public static int[] BubbleSort(int[] arr) {
        for (int i = 0; i < arr.length; i++) {  // i只起到计数作用
            for (int j = 0; j < arr.length - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                }
            }
        }
        return arr;
    }

    /*
    在i+1到最后，选出最小的，和i交换
    不稳定，比如7 7 1，第一个7和1交换，到第二个7后面
     */
    public static int[] SelectionSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            int tmp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = tmp;
        }
        return arr;
    }

    /*
    假设前i个数有序，然后把第i+1个数插入到前i个数里面，顺序后移
    稳定
     */
    public static int[] InsertionSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = i + 1; j > 0; j--) {
                if (arr[j] < arr[j - 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j - 1];
                    arr[j - 1] = tmp;
                } else {
                    break;
                }
            }
        }
        return arr;
    }

    /*
    类似归并排序 加上 插入排序
    不稳定
     */
    public static int[] ShellSort(int[] arr) {
        for (int gap = arr.length / 2; gap > 0; gap /= 2) {
            for (int i = 0; i < gap; i++) {
                for (int j = i + gap; j < arr.length; j += gap) {
                    if (arr[j] < arr[j - gap]) {
                        int tmp = arr[j];
                        arr[j] = arr[j - gap];
                        arr[j - gap] = tmp;
                    }
                }
            }
        }
        return arr;
    }

    /*
    先自顶向下分裂，再自顶向上排序，排序时另外开辟空间，双指针操作，时间复杂度为O(n)
    稳定
     */
    public static int[] MergeSort(int[] arr) {
        return mergeSort(arr, 0, arr.length - 1);
    }

    private static int[] mergeSort(int[] arr, int low, int high) {
        int mid = (low + high) / 2;
        if (low < high) {
            mergeSort(arr, low, mid);
            mergeSort(arr, mid + 1, high);
            merge(arr, low, mid, high);
        }
        return arr;
    }

    /*
    先构造最大堆————对所有非叶子节点，将小的数一层层扔下去
    交换最大堆的头和尾，把最大的扔到后面去，不管它，然后把可能不是最大的头一层层扔下去
    不稳定
     */
    private static void merge(int[] arr, int low, int mid, int high) {
        int[] temp = new int[high - low + 1];
        int i = low;    // 左指针
        int j = mid + 1;    // 右指针
        int k = 0;
        while (i <= mid && j <= high) {
            if (arr[i] < arr[j]) temp[k++] = arr[i++];
            else temp[k++] = arr[j++];
        }
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= high) temp[k++] = arr[j++];
        for (int l = 0; l < temp.length; l++) {
            arr[l + low] = temp[l];
        }
    }

    public static int[] HeapSort(int[] arr) {
        // 构造最大堆
        for (int i = arr.length / 2; i >= 0; i--) {
            minHeapDown(arr, i, arr.length);
        }
        for (int j = arr.length - 1; j > 0; j--) {
            int tmp = arr[j];
            arr[j] = arr[0];
            arr[0] = tmp;
            minHeapDown(arr, 0, j);
        }
        return arr;
    }

    // 将非叶子节点的小数沉下去
    private static void minHeapDown(int[] arr, int start, int end) {
        int tmp = arr[start];
        for (int k = start * 2 + 1; k < end; k = k * 2 + 1) {//从i结点的左子结点开始，也就是2i+1处开始
            if (k + 1 < end && arr[k] < arr[k + 1]) {//如果左子结点小于右子结点，k指向右子结点
                k++;
            }
            if (arr[k] > tmp) {//如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
                arr[start] = arr[k];
                start = k;
            } else {
                break;
            }
        }
        arr[start] = tmp;
    }

    /*
    自顶向下的partition
    不稳定，有多个a[j]的值时，不稳定发生在中枢元素和a[j]交换的时刻。
     */
    public static int[] QuickSort(int[] arr) {
        QuickSort(arr, 0, arr.length - 1);
        return arr;
    }

    private static void QuickSort(int[] arr, int lo, int hi) {
        if (lo >= hi) return;
        int index = partition(arr, lo, hi);
        QuickSort(arr, lo, index - 1);
        QuickSort(arr, index + 1, hi);
    }

    private static int partition(int[] arr, int lo, int hi) {
        int key = arr[lo];
        while (lo < hi) {
            while (arr[hi] >= key && lo < hi) hi--;
            arr[lo] = arr[hi];
            while (arr[lo] <= key && lo < hi) lo++;
            arr[hi] = arr[lo];
        }
        arr[lo] = key;
        return lo;
    }

    /*
    统计arr的最大最小值，另立一个数组记录每个元素出现的个数
     */
    public static int[] CountingSort(int[] arr) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        // 找出数组中的最大最小值
        for (int i = 0; i < arr.length; i++) {
            max = Math.max(max, arr[i]);
            min = Math.min(min, arr[i]);
        }
        int[] help = new int[max - min + 1];

        for (int i = 0; i < arr.length; i++) {
            help[arr[i] - min]++;
        }
        int index = 0;
        for (int i = 0; i < help.length; i++) {
            while (help[i]-- > 0) arr[index++] = i + min;
        }
        return arr;
    }

    /*
    将一定范围内的数映射到一个桶里
     */
    public static int[] BucketSort(int[] arr) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < arr.length; i++) {
            max = Math.max(max, arr[i]);
            min = Math.min(min, arr[i]);
        }

        //桶数
        int bucketNum = (max - min) / arr.length + 1;
        List<List<Integer>> bucketArr = new ArrayList<>(bucketNum);
        for (int i = 0; i < bucketNum; i++) {
            bucketArr.add(new ArrayList<>());
        }

        //将每个元素放入桶
        for (int i = 0; i < arr.length; i++) {
            int num = (arr[i] - min) / (arr.length);
            bucketArr.get(num).add(arr[i]);
        }

        //对每个桶进行排序
        for (int i = 0; i < bucketArr.size(); i++) {
            Collections.sort(bucketArr.get(i));
        }

        int index = 0;
        for (List<Integer> l : bucketArr) {
            for (int i : l) {
                arr[index++] = i;
            }
        }
        return arr;
    }

    /*
    按照数字位数从小到大进行排序，先比个位，再比十位。。
     */
    public static int[] RadixSort(int[] arr) {
        int[][] array = new int[10][arr.length + 1];
        for (int i = 0; i < 10; i++) {
            array[i][0] = 0;// array[i][0]记录第i行数据的个数
        }
        int d = getMaxWeishu(arr);
        for (int pos = 1; pos <= d; pos++) {
            for (int i = 0; i < arr.length; i++) {// 分配过程
                int row = getNumInPos(arr[i], pos);
                int col = ++array[row][0];
                array[row][col] = arr[i];
            }
            for (int row = 0, i = 0; row < 10; row++) {// 收集过程
                for (int col = 1; col <= array[row][0]; col++) {
                    arr[i++] = array[row][col];
                }
                array[row][0] = 0;// 复位，下一个pos时还需使用
            }
        }
        return arr;
    }

    //pos=1表示个位，pos=2表示十位
    public static int getNumInPos(int num, int pos) {
        int tmp = 1;
        for (int i = 0; i < pos - 1; i++) {
            tmp *= 10;
        }
        return (num / tmp) % 10;
    }

    //求得最大位数d
    public static int getMaxWeishu(int[] a) {
        int max = a[0];
        for (int i = 0; i < a.length; i++) {
            if (a[i] > max)
                max = a[i];
        }
        int tmp = 1, d = 1;
        while (true) {
            tmp *= 10;
            if (max / tmp != 0) {
                d++;
            } else
                break;
        }
        return d;
    }
}
