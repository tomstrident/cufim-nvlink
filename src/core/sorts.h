
#include <vector>
#include <cstring>


// fast for small sets (<20)
template<typename T1, typename T2> inline
void insertion_sort(T1* arr, const T2 size)
{
  for(T2 it = 1 ; it < size ; ++it)
  {
    T1 tmp = arr[it];
    T2 jt = it;

    for( ; (jt > 0) && (tmp < arr[jt - 1]) ; --jt)
    {
      arr[jt] = arr[jt - 1];
    }

    arr[jt] = tmp;
  }
}

// ============================================================================
// sorting algorithms
template<typename T>
inline void pigeonhole_sort(const std::vector<T>& arr, const T minval, const T maxval)
{
  int range = maxval - minval + 1;
  std::vector<T> holes[range];

  for(int i = 0 ; i < arr.size() ; i++)
  {
    holes[arr[i]-minval].push_back(arr[i]);
  }

  int index = 0;
  for(int i = 0 ; i < range ; i++)
  {
    /*
    while(!hole[j].empty()) {
         arr[count] = *(hole[j].begin());
         hole[j].erase(hole[j].begin());
         count++;
    */
    std::vector<int>::iterator it;
    for(it = holes[i].begin() ; it != holes[i].end() ; ++it)
    {
      arr[index++]  = *it;
    }
  }
}
// ----------------------------------------------------------------------------
template<typename T>
inline void counting_sort(const std::vector<T>& arr)
{
  /*
  int output[size+1];
   int max = getMax(array, size);
   int count[max+1];     //create count array (max+1 number of elements)
   for(int i = 0; i<=max; i++)
      count[i] = 0;     //initialize count array to all zero
   for(int i = 1; i <=size; i++)
      count[array[i]]++;     //increase number count in count array.
   for(int i = 1; i<=max; i++)
      count[i] += count[i-1];     //find cumulative frequency
   for(int i = size; i>=1; i--) {
      output[count[array[i]]] = array[i];
      count[array[i]] -= 1; //decrease count for same numbers
   }
   for(int i = 1; i<=size; i++) {
      array[i] = output[i]; //store output array to main array
   }
  */

  /*
  int k = arr[0];
    for(int i = 0; i < n; i++){
        k = max(k, arr[i]);
    }
    int count[k] = {0};
    for(int i = 0; i < n; i++){
        count[arr[i]]++;
    }
    for(int i = 1; i <= k; i++){
        count[i]+=count[i-1];
    }
    int output[n];
    for(int i=n-1; i>=0; i--){
        output[--count[arr[i]]] = arr[i];
    }
    for(int i=0; i<n; i++){
        arr[i] = output[i];
    }
  */

  /*
  int output_array[s];
	int count_array[r];
	
	// initialize all elements to 0 in count array
	for(int i=0;i<r;i++)
		count_array[i]=0;
		
	// to take a count of all elements in the input array
	for(int i=0;i<s;i++)
		++count_array[input_array[i]];
	
	// cummulative count of count array to get the 
	// positions of elements to be stored in the output array
	for(int i=1;i<r;i++)
		count_array[i]=count_array[i]+count_array[i-1];
	
	// placing input array elements into output array in proper
	//  positions such that the result is a sorted array in ASC order
	for(int i=0;i<s;i++)
		output_array[--count_array[input_array[i]]] = input_array[i];
	
	// copy output array elements to input array
	for(int i=0;i<s;i++)
		input_array[i]=output_array[i];
  */

  /*
  int mi, mx, z = 0; findMinMax( arr, len, mi, mx );
	int nlen = ( mx - mi ) + 1; int* temp = new int[nlen];
	memset( temp, 0, nlen * sizeof( int ) );
 
	for( int i = 0; i < len; i++ ) temp[arr[i] - mi]++;
 
	for( int i = mi; i <= mx; i++ )
	{
	    while( temp[i - mi] )
	    {
		arr[z++] = i;
		temp[i - mi]--;
	    }
	}
 
	delete [] temp;
  */
  /*
  // The output character array
  // that will have sorted arr
  char output[strlen(arr)];

  // Create a count array to store count of individual
  // characters and initialize count array as 0
  int count[RANGE + 1], i;
  memset(count, 0, sizeof(count));

  // Store count of each character
  for (i = 0; arr[i]; ++i)
    ++count[arr[i]];

  // Change count[i] so that count[i] now contains actual
  // position of this character in output array
  for (i = 1; i <= RANGE; ++i)
    count[i] += count[i - 1];

  // Build the output character array
  for (i = 0; arr[i]; ++i) {
    output[count[arr[i]] - 1] = arr[i];
    --count[arr[i]];
  }*/

  /*
  For Stable algorithm
  for (i = sizeof(arr)-1; i>=0; --i)
  {
  output[count[arr[i]]-1] = arr[i];
  --count[arr[i]];
  }

  For Logic : See implementation
  */

  // Copy the output array to arr, so that arr now
  // contains sorted characters
  //for (i = 0; arr[i]; ++i)
  //  arr[i] = output[i];
}
// ----------------------------------------------------------------------------