The assignment was to use dataset from opus_books that too specific to english to french. The following steps were also required to complete the assignment:
1. Remove the English sentences with tokens more than 150.
2. Remove french sentences where len(fench_sentences) > len(english_sentrnce) + 10

3. Parameter Sharing.
One Cycle Policy. (Model trained for 40 epochs only).
Reducing the hidden layer in feed forward network from 1024 to 256.
Dynamic Padding.
