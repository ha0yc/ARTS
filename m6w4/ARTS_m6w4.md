# Algorithm
## Leetcode 456. 132 Pattern
* difficulty: Medium
* description: Given a sequence of n integers a1, a2, ..., an, a 132 pattern is a subsequence ai, aj, ak such that i < j < k and ai < ak < aj. Design an algorithm that takes a list of n numbers as input and checks whether there is a 132 pattern in the list.

* analyse:因为平时用java比较多，所以还是想在刷题的时候练一练python。其实在做这道题的时候没有什么头绪，开始先暴力解了一遍，大概是n的三次方的复杂度,但是肯定是TLE了。然后想了下优化的解法，先考虑优化到n的平方的复杂度。考虑binary search的方法，由于132pattern和找峰值的题目很类似，但是由于其中这三个数字不是连续排位的，所以不能简单适用。
* solution

    1. Solution 1 O(n*n)

        这个解法就是从左向右遍历，找到向上的趋势；然后从此位置再向右找到向下的趋势；找到向下的趋势后再向右找到介于这个区间的点。每个趋势开始的地方其实就是一个极点。每次遍历就是找到从左向右的所有的极小值，找到每个极小值之后都会向右找到右边区间所有的极大值，从而找到是否存在132Pattern。
        代码如下：
        ```python
            def find132Pattern(self, nums):
                """
                :type nums: List[int]
                :rtype: bool
                """
                length = len(nums)
                if length <= 2 :
                    return False
                
                i = 0
                j = 0
                k = 0
                while i < length - 1 :
                    while i < length - 1 and nums[i] >= nums[i + 1] :
                        i += 1
                    j = i + 1
                    while j < length - 1 and nums[j] <= nums[j + 1] :
                        j += 1
                    k = j + 1
                    while k < length :
                        if nums[i] < nums[k] and nums[k] < nums[j]:
                            return True
                        k += 1
                    i = j + 1

                return False
        ```
    2. Solution2 n*n
        
        这个解法就是找到每个点左边的最小值，可以这样看：当左边区间取最小值时，最小值和当前点组成的最小值-最大值对可以获得132pattern的概率最大。
        ```python
            def find132PatternWithMin(self, nums):
                """
                :type nums: List[int]
                :rtype: bool
                """
                length = len(nums)
                if length <= 2 :
                    return False
                minium = sys.maxint
                
                for j in range(length):
                    minium = min(minium, nums[j])
                    if minium == nums[j]:
                        continue
                    for k in range(j + 1, 1, length - 1):
                        if nums[k] > minium and nums[k] < nums[j] :
                            return True
        
                return False
        ```
    
    3. Solution 3 O(n)

        这个解法是看了discuss里面的一个根据栈的解法来做的（https://discuss.leetcode.com/topic/67881/single-pass-c-o-n-space-and-time-solution-8-lines-with-detailed-explanation）
        也就是，从右向左遍历，s1是132中的1，即遍历对应的值，也就是要找s1右边是否存在32.初始化s3为比s1小的最大值，从右向左，如果stack空或者s1比当前栈顶值大，则出栈并更新s3直到栈顶比s1大。也就是栈内存的是按顺序的降序的数字，同时栈内存储的都是比s3大的数字，而且每次遍历都保证stack非空。因此由于栈内总比s3要大，所以只要找到比s3小的s1就可以了。
    
        ```python
            def find132pattern(self, nums):
                """
                :type nums: List[int]
                :rtype: bool
                """
                length = len(nums)
                if length <= 2:
                        return False
                stack = []
                s3 = -sys.maxsize
                for i in range(length - 1, -1, -1):
                    print(s3)
                    if nums[i] < s3 :
                        return True
                    else:
                        while len(stack) != 0 and nums[i] > stack[-1] :    
                            s3 = stack[-1]
                            stack.pop()

                    stack.append(nums[i])

                return False 
        ```

