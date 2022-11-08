# cython: infer_types=True
cimport cython

# Cython memory pool for memory management
from cymem.cymem cimport Pool
from tqdm import tqdm

# C string management
from libc.string cimport memcpy           # C string copy function
from libc.stdint cimport uint32_t         # C unsigned 32-bit integer type

# Cython Hash table and fast counter
from preshed.maps cimport PreshMap        # Hash table
from preshed.counter cimport count_t      # Count type (equivalent to C int64_t)
from preshed.counter cimport PreshCounter # Fast counter

# spaCy C functions and types
from spacy.strings cimport hash_utf8      # Hash function (using MurmurHash2)
from spacy.typedefs cimport hash_t        # Hash type (equivalent to C uint64_t)
from spacy.strings cimport Utf8Str        # C char array/pointer
from spacy.strings cimport decode_Utf8Str # C char array/pointer to Python string function

cdef PreshMap global_hashmap = PreshMap(initial_size=1024)

cdef PreshCounter fast_count(list words):
    '''Count the number of occurrences of every word'''
    cdef:
        PreshCounter counter = PreshCounter(initial_size=256)
        bytes word
    for word in tqdm(words):
        # Insert the word into the hash table, and increment the counter with
        # the 64-bit hash
        counter.inc(_insert_in_hashmap(word, len(word),global_hashmap), 1)
 
    return counter


def text2bow(list words):
    '''Build the BoW representation of a list of words'''
    cdef:
        hash_t wordhash
        int i, freq
        list bow
    words = [word.encode("utf8") for word in words]
    # First count the number of occurrences of every word
    counter = fast_count(words)

    # Convert the PreshCounter object to a more readable Python list `bow`,
    # for further usage
    bow = []
    for i in range(counter.c_map.length):
        wordhash = counter.c_map.cells[i].key
        if wordhash != 0:
            freq = <count_t>counter.c_map.cells[i].value
            # We use the 64-bit hashes instead of integer ids, which works
            # as well
            bow.append((get_unicode(wordhash,global_hashmap), freq))
 
    return bow

def fast_encoding(
    list sentences,
    list bow,
    str output,
    int verbose
    )->list:
    cdef :
        PreshMap bow_hashmap = PreshMap(initial_size=1024)
        PreshMap freq_hashmap = PreshMap(initial_size=1024)
        bytes b_word
        int freq
        list sentence
        hash_t key
        int i
    bow = [(word.encode("utf8"),freq) for word, freq in bow]
    for i,(b_word,freq) in enumerate(bow):
        key = _insert_in_hashmap(b_word,len(b_word),bow_hashmap)
        if output == "int":
            _insert_in_freqmap(freq_hashmap,key,i)
        elif output == "count":
            _insert_in_freqmap(freq_hashmap,key,freq)
    sentences = [[word.encode("utf8") for word in sentence] for sentence in sentences]
    encoded_sents = []
    for sentence in tqdm(sentences,disable=verbose):
        encoded_sentence = []
        for b_word in sentence :
            encoded_word = encode(b_word,len(b_word),freq_hashmap)
            if encoded_word != -1:
                encoded_sentence.append(encoded_word)
        encoded_sents.append(encoded_sentence)
    return encoded_sents

cdef int encode(char* utf8_string, int length, PreshMap hashmap):
    cdef hash_t key = hash_utf8(utf8_string, length)
    cdef int* value = <int*>hashmap.get(key)
    if value is NULL:
        return -1
    else :
        return value[0]

cdef void _insert_in_freqmap(PreshMap freq_hashmap, hash_t key, int index):
    cdef int* value = <int*>freq_hashmap.get(key)
    if value is not NULL:
        pass
    else :
        value = <int*>freq_hashmap.mem.alloc(1,sizeof(int))
        value[0] = index
        freq_hashmap.set(key,value)

@cython.final
cdef hash_t _insert_in_hashmap(char* utf8_string, int length, PreshMap hashmap):
    cdef hash_t key = hash_utf8(utf8_string, length)
    cdef Utf8Str* value = <Utf8Str*>hashmap.get(key)
    if value is not NULL:
        return key
    value = _allocate(hashmap.mem, <unsigned char*>utf8_string, length)
    hashmap.set(key, value)
    return key


cdef Utf8Str* _allocate(Pool mem, const unsigned char* chars, uint32_t length) except *:
    cdef:
        int n_length_bytes
        int i
        Utf8Str* string = <Utf8Str*>mem.alloc(1, sizeof(Utf8Str))
        uint32_t ulength = length


    if length < sizeof(string.s):
        string.s[0] = <unsigned char>length
        memcpy(&string.s[1], chars, length)
        return string
    elif length < 255:
        string.p = <unsigned char*>mem.alloc(length + 1, sizeof(unsigned char))
        string.p[0] = length
        memcpy(&string.p[1], chars, length)
        return string
    else:
        i = 0
        n_length_bytes = (length // 255) + 1
        string.p = <unsigned char*>mem.alloc(length + n_length_bytes, sizeof(unsigned char))
        for i in range(n_length_bytes-1):
            string.p[i] = 255
        string.p[n_length_bytes-1] = length % 255
        memcpy(&string.p[n_length_bytes], chars, length)
    return string
 
 
cdef unicode get_unicode(hash_t wordhash, PreshMap hashmap):
    utf8str = <Utf8Str*>hashmap.get(wordhash)
    if utf8str is NULL:
        raise KeyError(f'{wordhash} not in hash table')
    else:
        return decode_Utf8Str(utf8str)

