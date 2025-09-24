---
title: "GGML - стартираме модели на всички устройства"
date: 2024-12-06
draft: false
description: "Разбор на първата версия на библиотеката GGML - изучаваме архитектурата, тензорите, паметта и основните примитиви"
tags: ["GGML", "LLM", "ML", "C", "тензори", "машинно обучение"]
categories: ["Разработка", "Машинно обучение"]
---

# GGML - LLM на всички устройства

## Съдържание

- [Какво е GGML](#какво-е-ggml)
- [Инсталация и стартиране](#инсталация-и-стартиране)
- [Основни примитиви](#основни-примитиви)
  - [Памет](#памет)
    - [Тензор](#тензор)
    - [Контекст](#контекст)
    - [Подравняване](#подравняване)
    - [Типове данни](#типове-данни)
    - [Оптимизации](#оптимизации)
  - [Тензори в паметта](#тензори-в-паметта)
    - [Проверка на размерността](#проверка-на-размерността)
    - [Проверка на смежност / padding](#проверка-на-смежност--padding)
    - [Проверка на смежност без първото измерение](#проверка-на-смежност-без-първото-измерение)
    - [Размер на тензорите](#размер-на-тензорите)
  - [Създаване на тензори](#създаване-на-тензори)
- [Автоматично диференциране и обучение](#автоматично-диференциране-и-обучение)
  - [Възможности на GGML за обучение](#възможности-на-ggml-за-обучение)
  - [Пример за обучение на линейна регресия](#пример-за-обучение-на-линейна-регресия)

## Какво е GGML

В момента PyTorch е най-популярният framework (заместил TensorFlow) за работа с ML. Основната задача на такива frameworks е работата с тензори (структури от данни, които представляват скалари, вектори или многомерни масиви). Библиотеките предоставят възможност за извършване на математически операции, намиране на градиенти (изчисляване на стойностите на производната на функцията в дадена точка), съдържат най-популярните алгоритми и позволяват използването на предимствата на устройствата, на които се използва библиотеката, например извършване на изчисления на GPU или използване на специални математически копроцесори или специфични CPU инструкции, които позволяват ускоряване на изчисленията на този тип устройства. Основният недостатък, когато се опитате да покриете всички сценарии, е размерът на framework-а и всички зависимости.

В противовес на всеобхватния подход, Георги Герганов от гр. София създаде библиотеката GGML (оттук и името: GG - инициалите на автора и ML). Идеята беше да се напише такава библиотека, която би позволила стартирането на LLM на голям брой устройства с МИНИМАЛНИ изисквания, като Raspberry Pi или слаби лаптопи. Тя позволява разделянето на изчисленията между CPU и GPU и компресирането (квантоването на моделите), като по този начин значително намалява размера на моделите, незначително губейки качество, но предоставяйки на потребителя възможността сам да решава какво да постави в приоритет: размер или качество.

В днешно време GGML поддържа изчисления на различни архитектури (NVIDIA, AMD, Apple Metal, ARM и т.н.). Но аз, вместо да се занимавам с всички детайли, искам да разгледам първата версия (първия commit от GitHub репозиторито) и да видя с какво всичко започна. В тази версия още нямаше поддръжка на GPU, но като основа за запознаване - това е отличен пример. Затова ако ви интересува как да използвате библиотеката днес, тази статия не е за вас.
Така че, нека видим с какво всичко започна и да се опитаме да разберем как работи всичко това и да стартираме проста модел. Хайде!

## Инсталация и стартиране

Клонираме проекта и се превключваме към първия commit

```bash
git clone https://github.com/ggml-org/ggml
cd ggml
git checkout fb558f
```

Целият код, без заглавните файлове - в един файл!!! : src/ggml.c

Нека опитаме да компилираме проекта, за да проверим че всичко е на място.

```sh
❯ cd ..
❯ mkdir build
❯ cd build
❯ cmake ../ggml/ -DCMAKE_POLICY_VERSION_MINIMUM=3.5
-- The C compiler identification is GNU 15.1.1
-- The CXX compiler identification is GNU 15.1.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.49.0")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- Configuring done (0.6s)
-- Generating done (0.0s)
-- Build files have been written to: /home/spgtty/edu/ggml/build
```

Компилираме

```bash
❯ make
[  4%] Building C object src/CMakeFiles/ggml.dir/ggml.c.o
[  8%] Linking C static library libggml.a
[  8%] Built target ggml
[ 12%] Building C object tests/CMakeFiles/test-vec0.dir/test-vec0.c.o
[ 16%] Linking C executable ../bin/test-vec0
[ 16%] Built target test-vec0
[ 20%] Building C object tests/CMakeFiles/test-vec1.dir/test-vec1.c.o
[ 25%] Linking C executable ../bin/test-vec1
[ 25%] Built target test-vec1
[ 29%] Building C object tests/CMakeFiles/test-grad0.dir/test-grad0.c.o
[ 33%] Linking C executable ../bin/test-grad0
[ 33%] Built target test-grad0
[ 37%] Building C object tests/CMakeFiles/test-mul-mat0.dir/test-mul-mat0.c.o
[ 41%] Linking C executable ../bin/test-mul-mat0
[ 41%] Built target test-mul-mat0
[ 45%] Building C object tests/CMakeFiles/test0.dir/test0.c.o
[ 50%] Linking C executable ../bin/test0
[ 50%] Built target test0
[ 54%] Building C object tests/CMakeFiles/test1.dir/test1.c.o
[ 58%] Linking C executable ../bin/test1
[ 58%] Built target test1
[ 62%] Building C object tests/CMakeFiles/test2.dir/test2.c.o
[ 66%] Linking C executable ../bin/test2
[ 66%] Built target test2
[ 70%] Building C object tests/CMakeFiles/test3.dir/test3.c.o
[ 75%] Linking C executable ../bin/test3
[ 75%] Built target test3
[ 79%] Building CXX object examples/CMakeFiles/ggml_utils.dir/utils.cpp.o
[ 83%] Linking CXX static library libggml_utils.a
[ 83%] Built target ggml_utils
[ 87%] Building CXX object examples/gpt-2/CMakeFiles/gpt-2.dir/main.cpp.o
[ 91%] Linking CXX executable ../../bin/gpt-2
[ 91%] Built target gpt-2
[ 95%] Building CXX object examples/gpt-j/CMakeFiles/gpt-j.dir/main.cpp.o
[100%] Linking CXX executable ../../bin/gpt-j
[100%] Built target gpt-j
```

Всичко е наред, дори в първия commit вече има примери за gpt-2 и gpt-j. Страхотно!

```
ggml/
├── include/ggml/ggml.h # Public API definitions
├── src/ggml.c # Core implementation
...
```

```bash
❯ cd ..
❯ mkdir playground
❯ ls
build  ggml  playground
❯ cd playground
❯ gcc main.c -I../ggml/include/ggml -L../build/src -lggml -lm -o main
```

## Основни примитиви

## Памет

### Тензор

Нека започнем със стандартния за всички ML библиотеки примитив - тензора. В нашия бъдещ граф на изчисленията - това са възлите.

```c
struct ggml_tensor {
	enum ggml_type type; // Data type (F32, F16, etc.)

	int n_dims; // Number of dimensions
	int ne[GGML_MAX_DIMS]; // Number of elements per dimension
	size_t nb[GGML_MAX_DIMS]; // Number of bytes (stride)
                              // nb[0] = sizeof(type)
                              // nb[1] = nb[0]   * ne[0] + padding
                              // nb[i] = nb[i-1] * ne[i-1]

	enum ggml_op op; // Operation that created this tensor
	bool is_param; // Is this a parameter (for gradients)

	struct ggml_tensor * grad; // Gradient tensor
	struct ggml_tensor * src0; // First source tensor
	struct ggml_tensor * src1; // Second source tensor

	void * data; // Actual data pointer
};
```

type - това е enum, означаващ типа данни,
n_dims - размерност (0-скалар, 1 вектор, 2 - матрица и т.н.)
ne - масив, съдържащ броя елементи в дадената размерност (например за матрица 3x4 - ne[0]=4 cols, ne[1]=3 rows)
nb - стъпка на размерността в байтове, т.е. през колко байта започва новия елемент от размерността. За матрица 3х4 (3 реда и 4 колони) с данни F32 (4 байта на променлива) получаваме че nb[0] = 4 (размера на float), nb[1] = 4х4=16 байта в реда. Ако елементите в реда не са подравнени, трябва да се добави padding.
Защо не можем просто да изчислим nb на базата на размера на типа данни и размерността? Това е направено, за да можем да работим с проекции на по-малки размерности в рамките на дадения тензор (т.е. представете си че имате картинка rgb и правите свъртка (convolution) - трябва ни да работим с kernel 16на16 в рамките на по-голям размер, няма да ни се налага да копираме данни, ще можем да използваме nb за да намираме следващия ред). Друг аспект е подравняването на данните, за да всеки нов ред започва от адрес кратен на `#define GGML_MEM_ALIGN 16`

op - enum описващ операцията във възела (0 - NOOP ако тензорът просто съдържа данни).
is_param - флаг определя дали трябва да се изчислява градиента в дадения възел (грубо казано съдържа ли тензорът X или θ, аналог на torch.no_grad())

след това следва указател към друг тензор, в който се записват стойностите на градиентите,
src0, src1 - родителските възли в нашия граф на изчисленията

data - указател към областта от паметта съдържаща данните

### Контекст

```c
struct ggml_context {
	size_t mem_size; // Total memory size
	void * mem_buffer; // Memory buffer

	bool mem_buffer_owned; // Whether we own the buffer
	int n_objects; // Number of allocated objects

	struct ggml_object * objects_begin; // Linked list of objects
	struct ggml_object * objects_end;
};
```

Контекстът в ggml е областта от паметта, в която работим. Идеята е да не заделяме памет под всеки тензор, като извикваме malloc много пъти, а вместо това да заделим голяма област от паметта и да работим в нея, назначавайки области от тази памет под тензорите.

Структурата е проста
size_t - размер на паметта
mem_buffer - адресът на началото на блока памет. Можем да го оставим празен, тогава ggml ще задели памет за нас с помощта на malloc при инициализацията

mem_buffer_owned - флаг показва кой е заделил паметта, ggml или ние, за да разберем кой трябва да я освободи

След това простo свързано дърво от ggml обекти,
n_objects
objects_begin
objects_end

```c
struct ggml_object {
    size_t offset;
    size_t size;

    struct ggml_object * next;

    char padding[8]; //Explicit padding (not used for data)
};
```

Всеки възел съдържа отместване спрямо паметта на контекста, където се намира нашия тензор. Така, за да преминем от обект към тензор, ни трябва:

```c
#define GGML_OBJECT_TO_TENSOR(obj) \
((struct ggml_tensor *) ((char *) ctx->mem_buffer + obj->offset))

struct ggml_object * obj = find_some_object(ctx);
struct ggml_tensor * tensor = GGML_OBJECT_TO_TENSOR(obj);

float * actual_data = (float *) tensor->data;
```

### Подравняване

Отделно си заслужава да разгледаме подравняването на данните. Тъй като библиотеката използва SIMD (Single Instruction Multiple Data) инструкции на процесора, което позволява за времето на една операция да се провежда тази операция паралелно върху масиви от данни, то това налага определени изисквания към адресирането на данните, а именно те трябва да бъдат подравнени, т.е. да се намират на адрес кратен на 16.

GGML реализира това така -

```c
// align to GGML_MEM_ALIGN
size_needed = ((size_needed + GGML_MEM_ALIGN - 1)/GGML_MEM_ALIGN)*GGML_MEM_ALIGN;
```

Ще разгледаме по-късно как се създава тензор, там се използва този код.

Подравняването ни позволява да бъдем сигурни че данните лежат в нужните ни адреси

```
Context Memory Buffer:
┌───────────────────────────────────────────────────────────┐
│ ctx->mem_buffer                                           │
├──────────┬──────────────┬──────────────────┬──────────────┤
│ object1  │ tensor1      │ data1 (aligned)  │ object2      │
│ metadata │ (aligned)    │ [1.0,2.0,3.0]    │ metadata     │
├──────────┼──────────────┼──────────────────┼──────────────┤
│ offset=0 │ offset=16    │ offset=64        │ offset=144   │
│          │ (padded)     │ (padded)         │ (padded)     │
└──────────┴──────────────┴──────────────────┴──────────────┘
```

### Типове данни

GGML поддържа следните типове данни

```c
enum ggml_type {
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_COUNT,
};
```

Особено внимание заслужава F16 - типа с половинна точност от стандартния Float 32. Но този тип не се поддържа в x86, затова всички изчисления над този тип първо превеждат данните в F32, а после обратно

```c
typedef uint16_t ggml_fp16_t; // Just a 16-bit integer!

#ifdef __ARM_NEON
// we use the built-in 16-bit float type
typedef __fp16 ggml_fp16_t;
#else
typedef uint16_t ggml_fp16_t;
#endif

float ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);

```

Представянето на FP16 се различава от представянето на FP32 само с броя битове за експонентата и мантисата:

```
// FP16 (16 bits total):
// ┌─┬─────────┬──────────┐
// │S│EEEEE    │MMMMMMMMMM│
// └─┴─────────┴──────────┘
//  1  5 bits    10 bits

// FP32 (32 bits total):
// ┌─┬────────────┬───────────────────────┐
// │S│EEEEEEEE    │MMMMMMMMMMMMMMMMMMMMMMM│
// └─┴────────────┴───────────────────────┘
//  1  8 bits          23 bits
```

В тази статия няма да разглеждам функциите за превод от един тип в друг, тъй като за това ми трябва да напиша още една статия. Не можем просто да преместим битовете, тъй като различните типове имат различна базова стойност на експонентата.
Също така трябва да се вземат предвид специалните нормализирани стойности.

Ето как изглежда реализацията на конвертирането:

```c
#ifdef __ARM_NEON

#include <arm_neon.h>
float ggml_fp16_to_fp32(ggml_fp16_t x) {
    return x;
}
ggml_fp16_t ggml_fp32_to_fp16(float x) {
    return x;
}

#else

#include <immintrin.h>
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = { w };
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32 = { f };
	return fp32.as_bits;
}

float ggml_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

ggml_fp16_t ggml_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
#endif

```

### Оптимизации

Интересно е, че дори в първия commit вече присъстват оптимизации. Това е много яко.

От това, което ми направи впечатление - това е padding-а за променливите, т.е. разделянето на променливите в паметта, за да се избегне копирането на кеша от ядро в ядро
`cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);`

Също така в кода присъстват SIMD оптимизации, за паралелно изчисляване на математически операции, например -

```c
...
#else // x86 AVX implementation
const int n32 = 32*(n/32); // Process 32 elements at a time

__m256 sum0 = _mm256_setzero_ps();
__m256 sum1 = _mm256_setzero_ps();
__m256 sum2 = _mm256_setzero_ps();
__m256 sum3 = _mm256_setzero_ps();

for (int i = 0; i < n32; i += 32) {
	// Convert F16 to F32 and load
	__m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 0)));
	__m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 0)));

	// Fused multiply-add
	sum0 = _mm256_fmadd_ps(x0, y0, sum0);
	// ... process remaining vectors
}
...
#endif
```

Няма да се спирам на оптимизациите, но трябва да се разбира, че благодарение на това библиотеката работи бързо, използвайки предимствата на платформата, на която е стартирана.

## Тензори в паметта

За удобство GGML има няколко полезни функции, използвани при изчисленията. По-долу са представени съкратени извадки от кода.

### Проверка на размерността

```c
// From ggml.c - determine tensor dimensionality
bool ggml_is_scalar(const struct ggml_tensor * tensor) {
	return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}
bool ggml_is_vector(const struct ggml_tensor * tensor) {
	return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}
bool ggml_is_matrix(const struct ggml_tensor * tensor) {
	return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

```

### Проверка на смежност / padding

```c

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
	return
		tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
		tensor->nb[1] == tensor->nb[0]*tensor->ne[0] &&
		tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
		tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

```

Функцията проверява

1. Елементите и редовете са плътно опаковани в паметта (няма padding-ове)
2. Всяка размерност умножена по броя елементи в нея - това е стъпката на следващата размерност.

### Проверка на смежност без първото измерение

```c
bool ggml_is_padded_1d(const struct ggml_tensor * tensor) {
	return
		tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
		tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
		tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}
```

Същото като по-горе, но без проверка на padding-а по първото измерение.

Защо ни трябва такава функция?

- **SIMD операции**: Изискват `ggml_is_contiguous() == true`
- Операции по редове: Достатъчно е `ggml_is_padded_1d() == true`
- Произволни операции: Трябва да работят при всякакъв padding

НО - в първата версия, padding-ът за редовете отсъства в имплементацията за създаване на нов тензор, затова тази и предишната функция винаги връщат true, освен ако не създавате тензори сами.

### Размер на тензорите

Коментарите са излишни, кодът е ясен

```c
int ggml_nelements(const struct ggml_tensor * tensor) {
	return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int ggml_nrows(const struct ggml_tensor * tensor) {
	return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
	return ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type];
}
```

## Създаване на тензори

Преди да преминем към разглеждането на имплементацията на функцията за създаване на тензори, нека опитаме да напишем няколко примера и да проверим че всичко работи както е замислено.

<details>
<summary><b>Пример за работа с тензори</b></summary>

```c
#include <stdio.h>
#include <ggml.h>

void print_tensor_info(const struct ggml_tensor* t, const char* name) {
    printf("\n=== %s ===\n", name);
    printf("Type: %d, Dims: %d\n", t->type, t->n_dims);
    printf("Shape (ne): [%d, %d, %d, %d]\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("Strides (nb): [%zu, %zu, %zu, %zu]\n", t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    printf("Elements: %d\n", ggml_nelements(t));
    printf("Rows: %d\n", ggml_nrows(t));
    printf("Bytes: %zu\n", ggml_nbytes(t));
    printf("Properties:\n");
    printf("  Scalar: %s\n", ggml_is_scalar(t) ? "YES" : "NO");
    printf("  Vector: %s\n", ggml_is_vector(t) ? "YES" : "NO");
    printf("  Matrix: %s\n", ggml_is_matrix(t) ? "YES" : "NO");
    printf("  Contiguous: %s\n", ggml_is_contiguous(t) ? "YES" : "NO");
    printf("  Padded 1D: %s\n", ggml_is_padded_1d(t) ? "YES" : "NO");
}

int main() {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 10, // 10 MB
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create different tensor shapes
    struct ggml_tensor* scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor* vector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
    struct ggml_tensor* tensor3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 28, 28, 64);
    struct ggml_tensor* tensor4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 224, 224, 3, 16);

    print_tensor_info(scalar, "Scalar");
    print_tensor_info(vector, "Vector");
    print_tensor_info(matrix, "Matrix");
    print_tensor_info(tensor3d, "3D Tensor");
    print_tensor_info(tensor4d, "4D Tensor (Batch)");

    ggml_free(ctx);
    return 0;
}

```

```
=== Scalar ===
Type: 4, Dims: 1
Shape (ne): [1, 1, 1, 1]
Strides (nb): [4, 4, 4, 4]
Elements: 1
Rows: 1
Bytes: 4
Properties:
  Scalar: YES
  Vector: YES
  Matrix: YES
  Contiguous: YES
  Padded 1D: YES

=== Vector ===
Type: 4, Dims: 1
Shape (ne): [100, 1, 1, 1]
Strides (nb): [4, 400, 400, 400]
Elements: 100
Rows: 1
Bytes: 400
Properties:
  Scalar: NO
  Vector: YES
  Matrix: YES
  Contiguous: YES
  Padded 1D: YES

=== Matrix ===
Type: 4, Dims: 2
Shape (ne): [64, 32, 1, 1]
Strides (nb): [4, 256, 8192, 8192]
Elements: 2048
Rows: 32
Bytes: 8192
Properties:
  Scalar: NO
  Vector: NO
  Matrix: YES
  Contiguous: YES
  Padded 1D: YES

=== 3D Tensor ===
Type: 3, Dims: 3
Shape (ne): [28, 28, 64, 1]
Strides (nb): [2, 56, 1568, 100352]
Elements: 50176
Rows: 1792
Bytes: 100352
Properties:
  Scalar: NO
  Vector: NO
  Matrix: NO
  Contiguous: YES
  Padded 1D: YES

=== 4D Tensor (Batch) ===
Type: 4, Dims: 4
Shape (ne): [224, 224, 3, 16]
Strides (nb): [4, 896, 200704, 602112]
Elements: 2408448
Rows: 10752
Bytes: 9633792
Properties:
  Scalar: NO
  Vector: NO
  Matrix: NO
  Contiguous: YES
  Padded 1D: YES
ggml_free: context 0 with 5 objects has been freed. memory used = 9743552
```

</details>

<br/>

Нека отобразим паметта

<details>
<summary><b>Визуализация на паметта на тензора</b></summary>

```c
#include <stdio.h>
#include <stdlib.h>
#include <ggml.h>

void visualize_2d_layout(struct ggml_tensor* t) {
    if (!ggml_is_matrix(t)) {
        printf("Not a 2D matrix!\n");
        return;
    }

    printf("\n=== 2D Memory Layout ===\n");
    printf("Shape: %dx%d (width x height)\n", t->ne[0], t->ne[1]);
    printf("Strides: [%zu, %zu] bytes\n", t->nb[0], t->nb[1]);

    // Fill with sample data
    if (t->type == GGML_TYPE_F32) {
        float* data = (float*)t->data;
        for (int i = 0; i < ggml_nelements(t); i++) {
            data[i] = i;
        }

        // Print logical layout
        printf("\nLogical view:\n");
        for (int row = 0; row < t->ne[1]; row++) {
            printf("Row %d: ", row);
            for (int col = 0; col < t->ne[0]; col++) {
                // Calculate memory offset
                size_t offset = row * t->nb[1] + col * t->nb[0];
                float* element = (float*)((char*)t->data + offset);
                printf("%6.0f ", *element);
            }
            printf("\n");
        }

        // Print memory addresses
        printf("\nMemory addresses:\n");
        for (int row = 0; row < t->ne[1]; row++) {
            printf("Row %d: ", row);
            for (int col = 0; col < t->ne[0]; col++) {
                size_t offset = row * t->nb[1] + col * t->nb[0];
                printf("%6zu ", offset);
            }
            printf("\n");
        }
    }
}

int main() {
    struct ggml_init_params params = { .mem_size = 1024 * 1024, .mem_buffer = NULL };
    struct ggml_context* ctx = ggml_init(params);

    // Create a small matrix for visualization
    struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    visualize_2d_layout(matrix);

    ggml_free(ctx);
    return 0;
}

```

```
ggml_init: found unused context 0

=== 2D Memory Layout ===
Shape: 4x3 (width x height)
Strides: [4, 16] bytes

Logical view:
Row 0:      0      1      2      3
Row 1:      4      5      6      7
Row 2:      8      9     10     11

Memory addresses:
Row 0:      0      4      8     12
Row 1:     16     20     24     28
Row 2:     32     36     40     44
ggml_free: context 0 with 1 objects has been freed. memory used = 208
```

</details>

<br/>

# Операции върху тензори

Разборът на целия изходен код би отнел много време, но тъй като той е доста прост - ще го оставя тук в сгъваем блок, ако ви интересува. Кодът поддържа SIMD чрез Arm Neon архитектура (на Mac-а) или AVX на х86

<details>
<summary><b>Изходен код на операциите</b></summary>

```c
inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i] = x[i] + y[i]; }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] += x[i]; }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float v) { for (int i = 0; i < n; ++i) y[i] += v; }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i] = x[i] - y[i]; }
inline static void ggml_vec_set_f32 (const int n, float * x, const float v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = -x[i]; }
inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i] = x[i]*y[i]; }
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i] = x[i]/y[i]; }

inline static void ggml_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
	for (int i = 0; i < n; ++i) {
		y[i] += x[i]*v;
	}
}
inline static void ggml_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y) {

	ggml_float sum = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += x[i]*y[i];
	}
	*s = sum;
}

// ... (SIMD оптимизации са пропуснати за кратност)
```

</details>

<br/>

Този код в библиотеката фактически извършва всички операции, всички останали операции за работа с матрици и тензори ще използват кода за работа с вектори в крайна сметка.

Ето например, функцията за изчисляване на абсолютната стойност на тензор, която използва векторни операции за работа с данните.

```c
// ggml_compute_forward_abs

void ggml_compute_forward_abs_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_abs_f32(nc, // <-- абсолютна стойност на вектора
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}
```

# Изчислителен граф

GGML използва изчислителен граф за организация на операциите върху тензорите. Графът се състои от възли, където всеки възел представлява операция върху тензори. Възлите могат да бъдат свързани помежду си, образувайки верига от изчисления.

```c
// computation graph
struct ggml_cgraph {
    int n_nodes;
    int n_leafs;
    int n_threads;

    size_t work_size;
    struct ggml_tensor * work;

    struct ggml_tensor * nodes[GGML_MAX_NODES];
    struct ggml_tensor * grads[GGML_MAX_NODES];
    struct ggml_tensor * leafs[GGML_MAX_NODES];
    ...
};
```

Графът съдържа масиви от възли, градиенти и листа, както и размера на буфера за работа. Всеки възел в графа представлява тензор, който може да бъде резултат от операция или вход за други операции.

Също така заслужава внимание функцията за изчисляване на графа, която стартира нишки за изпълнение на операциите върху тензорите (ако е нужно разделяне на нишки). Тук е важно да се разбере, че възлите на графа са топологически подредени, и всеки възел може да зависи от резултатите на предишни възли. Затова, при изчисляване на графа, първо се изпълняват възлите, от които зависят други възли, а после се изпълняват възлите, които зависят от тях.

# Автоматично диференциране и обучение

GGML поддържа автоматично диференциране и вградени оптимизатори, което позволява обучение на модели без ръчно изчисляване на градиенти.

## Възможности на GGML за обучение

- **Автоматично диференциране**: GGML автоматично изчислява градиенти
- **Вградени оптимизатори**: Adam, L-BFGS и други алгоритми за оптимизация
- **Управление на паметта**: Автоматично управление на графите от изчисления
- **Обратно разпространение**: Поддръжка на backward pass за градиенти

## Пример за обучение на линейна регресия

Прост пример за обучение на линеен модел с помощта на вградения оптимизатор на GGML:

<details>
<summary><b>Пример за обучение на линейна регресия</b></summary>

```c
#include <ggml.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

bool is_close(float a, float b, float epsilon)
{
    return fabs(a - b) < epsilon;
}

int main()
{
    printf("=== GGML Simple Linear Regression Test ===\n\n");

    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = NULL,
    };

    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);
    opt_params.adam.alpha = 0.01f;

    printf("=== Simple Linear Regression Test ===\n");
    printf("Trying to fit: y = t0 + t1*x to polynomial data\n\n");

    // Create training data from polynomial y = x^2 + 2*x + 1
    const float xi[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float yi[5];

    for (int i = 0; i < 5; i++)
    {
        yi[i] = xi[i] * xi[i] + 2.0f * xi[i] + 1.0f; // y = x^2 + 2*x + 1
    }

    const int n = sizeof(xi) / sizeof(xi[0]);

    printf("Training data:\n");
    for (int i = 0; i < n; i++)
    {
        printf("x=%.2f -> y=%.2f\n", xi[i], yi[i]);
    }

    struct ggml_context *ctx0 = ggml_init(params);

    struct ggml_tensor *x = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n);
    struct ggml_tensor *y = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n);

    for (int i = 0; i < n; i++)
    {
        ((float *)x->data)[i] = xi[i];
        ((float *)y->data)[i] = yi[i];
    }

    struct ggml_tensor *t0 = ggml_new_f32(ctx0, 0.0f); // bias
    struct ggml_tensor *t1 = ggml_new_f32(ctx0, 0.1f); // weight

    // initialize auto-diff parameters:
    ggml_set_param(ctx0, t0);
    ggml_set_param(ctx0, t1);

    printf("Initial parameters: t0(bias)=%.4f, t1(weight)=%.4f\n",
           ggml_get_f32_1d(t0, 0), ggml_get_f32_1d(t1, 0));

    // f = sum_i[(t0 + t1*x_i - y_i)^2]/(2n)
    struct ggml_tensor *f =
        ggml_div(ctx0,
                 ggml_sum(ctx0,
                          ggml_sqr(ctx0,
                                   ggml_sub(ctx0,
                                            ggml_add(ctx0,
                                                     ggml_mul(ctx0, x, ggml_repeat(ctx0, t1, x)),
                                                     ggml_repeat(ctx0, t0, x)),
                                            y))),
                 ggml_new_f32(ctx0, 2.0f * n));

    // Check initial predictions and loss
    struct ggml_cgraph gf_init = ggml_build_forward(f);
    ggml_graph_compute(ctx0, &gf_init);

    printf("Initial loss: %.6f\n", ggml_get_f32_1d(f, 0));

    // Compute initial predictions manually to show them
    printf("\nInitial predictions:\n");
    for (int i = 0; i < n; i++)
    {
        float pred = ggml_get_f32_1d(t0, 0) + ggml_get_f32_1d(t1, 0) * xi[i];
        printf("x=%.2f: pred=%.3f, true=%.2f\n", xi[i], pred, yi[i]);
    }

    printf("\nTraining...\n");
    enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

    printf("Training result: %s\n",
           res == GGML_OPT_OK ? "SUCCESS" : res == GGML_OPT_DID_NOT_CONVERGE ? "DID_NOT_CONVERGE"
                                                                             : "OTHER");

    printf("\nFinal parameters: t0(bias)=%.4f, t1(weight)=%.4f\n",
           ggml_get_f32_1d(t0, 0), ggml_get_f32_1d(t1, 0));

    // Check final loss
    struct ggml_cgraph gf_final = ggml_build_forward(f);
    ggml_graph_compute(ctx0, &gf_final);
    printf("Final loss: %.6f\n", ggml_get_f32_1d(f, 0));

    // Show final predictions
    printf("\nFinal predictions:\n");
    float total_error = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float pred = ggml_get_f32_1d(t0, 0) + ggml_get_f32_1d(t1, 0) * xi[i];
        float error = fabsf(pred - yi[i]);
        total_error += error;
        printf("x=%.2f: pred=%.3f, true=%.2f, error=%.3f\n", xi[i], pred, yi[i], error);
    }
    printf("Average error: %.4f\n", total_error / n);

    ggml_free(ctx0);
    return 0;
}
```

</details>

<br/>

**Особености на примера:**

- **Простота**: Линейната регресия е една от най-простите и надеждни задачи в ML
- **Гарантирана сходимост**: Изпъкнала задача за оптимизация винаги се сходи
- **Разбираем резултат**: Лесно за проверка на правилността на обучението
- **Минимални изисквания**: Използва само основни операции на GGML

**Предимства на вградения оптимизатор на GGML:**

- Автоматично управление на паметта на графите
- Оптимизирани алгоритми (Adam, L-BFGS)
- Отсъствие на проблеми с препълване на стека
- По-малко код и по-лесен за използване
