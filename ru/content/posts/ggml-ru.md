---
title: "GGML - запускаем модели на всех устройствах"
date: 2024-12-06
draft: false
description: "Разбор первой версии библиотеки GGML - изучаем архитектуру, тензоры, память и основные примитивы"
tags: ["GGML", "LLM", "ML", "C", "тензоры", "машинное обучение"]
categories: ["Разработка", "Машинное обучение"]
---
# GGML - LLM на любых устройствах

## Содержание

- [Что такое GGML](#что-такое-ggml)
- [Установка и запуск](#установка-и-запуск)
- [Основные примитивы](#основные-примитивы)
  - [Память](#память)
    - [Тензор](#тензор)
    - [Контекст](#контекст)
    - [Выравнивание](#выравнивание)
    - [Типы данных](#типы-данных)
    - [Оптимизации](#оптимизации)
  - [Тензоры в памяти](#тензоры-в-памяти)
    - [Проверка размерности](#проверка-размерности)
    - [Проверка смежности / паддинга](#проверка-смежности--паддинга)
    - [Проверка смежности без первой размерности](#проверка-смежности-без-первой-размерности)
    - [Размер тензоров](#размер-тензоров)
  - [Создание тензоров](#создание-тензоров)
- [Автоматическое дифференцирование и обучение](#автоматическое-дифференцирование-и-обучение)
  - [Возможности GGML для обучения](#возможности-ggml-для-обучения)
  - [Пример обучения линейной регрессии](#пример-обучения-линейной-регрессии)

## Что такое GGML

На данный момент времени PyTorch является наиболее популярным фреймворком (заместившим TensorFlow) для работы с ML. Основной задачей таких фреймворков является работа с тензорами (структурами данных, которые представляют собой скаляры, векторы или многомерные массивы). Библиотеки предоставляют возможность производить математические операции, находить градиенты (вычислять значения производной функции в заданной точке), содержат наиболее популярные алгоритмы и при этом всём позволяют использовать преимущества устройств, на которых используется библиотека, например производить вычисления на GPU или использовать специальные математические копроцессоры или специфические инструкции CPU, которые позволяют ускорить вычисления на данном типе устройств. Основным недостатком, когда вы стараетесь охватить все сценарии, является размер фреймворка и всех зависимостей.

В противовес всеохватывающему подходу Георгий Герганов из г. София создал библиотеку GGML (отсюда и название: GG - инициалы автора и ML). Идея была написать такую библиотеку, которая позволяла бы запускать LLM на большом количестве устройств с МИНИМАЛЬНЫМИ требованиями, таких как Raspberry Pi или слабенькие ноутбуки. Она позволяет разделить вычисления между CPU и GPU и сжимать (квантовать модели), тем самым значительно снижая размер моделей, незначительно теряя качество, при этом предоставляя пользователю самому решать, что ставить в приоритет: размер или качество.

На сегодняшний день GGML поддерживает вычисления на разных архитектурах (NVIDIA, AMD, Apple Metal, ARM и т.д.). Но я же, вместо того, чтобы разбираться во всех деталях, хочу рассмотреть первую версию (первый коммит из GitHub репозитория) и посмотреть с чего же всё начиналось. В этой версии еще не было поддержки GPU, но как основа для знакомства - это отличный пример. Поэтому если вас интересует как пользоваться библиотекой на сегодняшний день, то эта статья не для вас.
Итак, давайте же посмотрим с чего всё начиналось, и попробуем разобраться как же всё это работает и запустить простенькую модель. Поехали!

## Установка и запуск

Склонируем проект, и переключимся на самый первый коммит

```bash
git clone https://github.com/ggml-org/ggml
cd ggml
git checkout fb558f
```

Весь код, исключая заголовки - в одном файле!!! : src/ggml.c

Давайте попробуем собрать проект, чтобы проверить что всё на месте.

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

Компилируем

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

Всё в порядке, даже в первом коммите уже есть примеры gpt-2 и gpt-j. Круто!

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

## Основные примитивы

## Память

### Тензор

Давайте начнём со стандартного для всех библиотек ML примитива - тензора. В нашем будущем графе вычислений - это узлы.

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

type - это enum означающий тип данных,
n_dims - размерность (0-скаляр, 1 вектор, 2 - матрица и тд)
ne - массив содержащий количество элементов в данной размерности (например для матрицы 3x4 - ne[0]=4 cols, ne[1]=3 rows)
nb - шаг размерности в байтах, то есть через сколько байтов начинается новый элемент размерности. Для матрицы 3х4 (3 строки и 4 колонки) с данными F32 (4 байта на переменную) мы получаем что nb[0] = 4 (размер float), nb[1] = 4х4=16 байтов в строке. Если элементы в строке не выравнены, то нужно добавить паддинг.
Почему мы не можем просто считать nb исходя из размера типа данных и размерности? Это сделано для того чтобы можно было работать с проекциями меньших размерностей внутри данного тензора (т.е. представьте если у вас есть картинка rgb и мы делаем свёртку (convolution) - нам нужно работать с кернелом 16на16 внутри большего размера, нам не нужно будет копировать данные, мы сможем использовать nb чтобы находить следующую строку). Другой аспект - это выравнивание данных, чтобы каждая новая строка начиналась с адреса кратного `#define GGML_MEM_ALIGN 16`

op - enum описывающий операцию в узле (0 - NOOP если тензор просто содержит данные).
is_param - флаг определяет нужно ли рассчитывать градиент в данном узле (грубо говоря содержит ли тензор X или \theta , аналог torch.no_grad())

далее идёт ссылка на другой тензор в который записываются значения градиентов,
src0, src1 - родительские узлы в нашем графе вычислений

data - ссылка на область памяти содержащей данные

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

Контекст в ggml - область памяти в которой мы работаем. Идея в том, чтобы не выделять память под каждый тензор вызывая malloc много раз, а вместо этого выделить большую область памяти и работать в ней, назначая области этой памяти под тензоры.

Структура проста
size_t - размер памяти
mem_buffer - адрес на начало блока памяти. Мы можем оставить пустым, тогда ggml выделит память для нас с помощью malloc при инициализации

mem_buffer_owned - флаг показывает кто выделил память, ggml или мы, чтобы понять кто должен её освобождать

Далее простое связное дерево ggml объектов,
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

Каждый узел содержит смещение относительно памяти контекста где находится наш тензор. Таким образом чтобы прийти от объекта к тензору нам нужно:

```c
#define GGML_OBJECT_TO_TENSOR(obj) \
((struct ggml_tensor *) ((char *) ctx->mem_buffer + obj->offset))

struct ggml_object * obj = find_some_object(ctx);
struct ggml_tensor * tensor = GGML_OBJECT_TO_TENSOR(obj);

float * actual_data = (float *) tensor->data;
```

### Выравнивание

Отдельно стоит рассмотреть выравнивание данных. Так как библиотека использует SIMD (Single Instruction Multiple Data) инструкции процессора, что позволяет за время одной операции проводить эту операцию параллельно на массивах данных, то это накладывает определенные требования к адресации данных, а именно они должны быть выровнены, тоесть располагаться по адресу кратному 16.

GGML реализует это так -

```c
// align to GGML_MEM_ALIGN
size_needed = ((size_needed + GGML_MEM_ALIGN - 1)/GGML_MEM_ALIGN)*GGML_MEM_ALIGN;
```

Мы рассмотрим позже как создаётся тензор, там и используется этот код.

Выравнивание позволяет нам быть уверенными что данные лежат в нужных нам адресах

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

### Типы данных

GGML поддерживает следующие типы данных

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

Особое внимание стоит уделить F16 - тип с половинной точностью от стандартного Float 32. Но этот тип не поддерживается в x86, поэтому все вычисления над этим типом сначала переводят данные в F32, а потом назад

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

Представление FP16 отличается от представления FP32 только количеством бит под экспоненту и мантиссу:

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

В этой статье я не буду рассматривать функции перевода одного типа в другой, так как для этого мне понадобится написать еще одну статью. Мы не можем просто переместить биты, так как у разных типов разное базовое значение экспоненты.
Еще нужно учитывать специальные нормализованные значения.

Вот как выглядит реализация конвертации:

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

Интересно, что даже в первом коммите уже присутствуют оптимизации. Это очень круто.

Из того, что бросилось в глаза - это паддинг для переменных, т.е. разделение переменных в памяти, чтобы избежать копирования кеша из ядра в ядро
`cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);`

Также в коде присутствуют SIMD оптимизации, для параллельного вычисления математических операций, например -

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

Я не буду останавливаться на оптимизациях, но нужно понимать, что благодаря этому библиотека работает быстро, используя преимущества платформы на которой запущена.

## Тензоры в памяти

Для удобства GGML имеет несколько полезных функций, используемых при вычислениях. Ниже представлены урезанные выборки из кода.

### Проверка размерности

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

### Проверка смежности / паддинга

```c

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
	return
		tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
		tensor->nb[1] == tensor->nb[0]*tensor->ne[0] &&
		tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
		tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

```

Функция проверяет

1. Элементы и строки плотно упакованы в памяти (нет паддингов)
2. Каждая размерность умноженная на количество элементов в ней - это шаг следующей размерности.

### Проверка смежности без первой размерности

```c
bool ggml_is_padded_1d(const struct ggml_tensor * tensor) {
	return
		tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
		tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
		tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}
```

То же что и выше, но без проверки паддинга по первой размерности.

Зачем нужна такая функция?

- **SIMD операции**: Требуют `ggml_is_contiguous() == true`
- Построчные операции: Достаточно `ggml_is_padded_1d() == true`
- Произвольные операции: Должны работать при любом паддинге

НО - в первой версии, паддинг для строк отсутствует в имплементации создания нового тензора, поэтому эта и предыдущая функция всегда возвращает true если только, вы не создаёте тензоры сами.

### Размер тензоров

Комментарии излишни, код понятен

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

## Создание тензоров

Перед тем как перейти к рассмотрению имплементации функции по созданию тензоров давайте попробуем написать пару примеров и проверить что всё работает как задумано.

<details>
<summary><b>Пример работы с тензорами</b></summary>

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

Давайте отобразим память

<details>
<summary><b>Визуализация памяти тензора</b></summary>

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

# Операции над тензорами

Разбирать весь исходный код, заняло бы много времени, но так как он довольно прост - я всё же оставлю его здесь в сворачиваемом блоке, если интересно. Код поддерживает SIMD через Arm Neon архитектуру (на маке) или AVX на х86

<details>
<summary><b>Исходный код операций</b></summary>

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

inline static void ggml_vec_dot_f16(const int n, float * restrict s, ggml_fp16_t * restrict x, ggml_fp16_t * restrict y) {

ggml_float sumf = 0.0;

#ifdef __ARM_NEON

const int n64 = 64*(n/64);

float16x8_t sum0 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum1 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum2 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum3 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum4 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum5 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum6 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float16x8_t sum7 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

float16x8_t x0, x1, x2, x3, x4, x5, x6, x7;
float16x8_t y0, y1, y2, y3, y4, y5, y6, y7;

for (int i = 0; i < n64; i += 64) {
	x0 = vld1q_f16(x + i + 0 );
	x1 = vld1q_f16(x + i + 8 );
	x2 = vld1q_f16(x + i + 16);
	x3 = vld1q_f16(x + i + 24);
	x4 = vld1q_f16(x + i + 32);
	x5 = vld1q_f16(x + i + 40);
	x6 = vld1q_f16(x + i + 48);
	x7 = vld1q_f16(x + i + 56);

	y0 = vld1q_f16(y + i + 0 );
	y1 = vld1q_f16(y + i + 8 );
	y2 = vld1q_f16(y + i + 16);
	y3 = vld1q_f16(y + i + 24);
	y4 = vld1q_f16(y + i + 32);
	y5 = vld1q_f16(y + i + 40);
	y6 = vld1q_f16(y + i + 48);
	y7 = vld1q_f16(y + i + 56);

	sum0 = vfmaq_f16(sum0, x0, y0);
	sum1 = vfmaq_f16(sum1, x1, y1);
	sum2 = vfmaq_f16(sum2, x2, y2);
	sum3 = vfmaq_f16(sum3, x3, y3);
	sum4 = vfmaq_f16(sum4, x4, y4);
	sum5 = vfmaq_f16(sum5, x5, y5);
	sum6 = vfmaq_f16(sum6, x6, y6);
	sum7 = vfmaq_f16(sum7, x7, y7);
}
// TODO: F16 - better way to reduce this ?

float16x8_t sum = vaddq_f16(sum0, sum1);

sum = vaddq_f16(sum, sum2);
sum = vaddq_f16(sum, sum3);
sum = vaddq_f16(sum, sum4);
sum = vaddq_f16(sum, sum5);
sum = vaddq_f16(sum, sum6);
sum = vaddq_f16(sum, sum7);

sumf += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
// I think this somehow makes the inference worse .. not sure ?
//sum0 = vaddq_f16(sum0, sum1);
//sum2 = vaddq_f16(sum2, sum3);
//sum4 = vaddq_f16(sum4, sum5);
//sum6 = vaddq_f16(sum6, sum7);
//sum0 = vaddq_f16(sum0, sum2);
//sum4 = vaddq_f16(sum4, sum6);
//sum0 = vaddq_f16(sum0, sum4);
//for (int i = 0; i < 8; ++i) {
// sumf += sum0[i];
//}

// leftovers
for (int i = n64; i < n; ++i) {
	sumf += ggml_fp16_to_fp32(x[i])*ggml_fp16_to_fp32(y[i]);
}
#else
// AVX 256-bit (unroll 4)

const int n32 = 32*(n/32);

__m256 sum0 = _mm256_setzero_ps();
__m256 sum1 = _mm256_setzero_ps();
__m256 sum2 = _mm256_setzero_ps();
__m256 sum3 = _mm256_setzero_ps();

__m256 x0, x1, x2, x3;
__m256 y0, y1, y2, y3;


for (int i = 0; i < n32; i += 32) {
	x0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 0 )));
	x1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 8 )));
	x2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 16)));
	x3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 24)));

	y0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 0 )));
	y1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 8 )));
	y2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 16)));
	y3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 24)));

	sum0 = _mm256_fmadd_ps(x0, y0, sum0);
	sum1 = _mm256_fmadd_ps(x1, y1, sum1);
	sum2 = _mm256_fmadd_ps(x2, y2, sum2);
	sum3 = _mm256_fmadd_ps(x3, y3, sum3);
}

const __m256 sum01 = _mm256_add_ps(sum0, sum1);
const __m256 sum23 = _mm256_add_ps(sum2, sum3);
const __m256 sum0123 = _mm256_add_ps(sum01, sum23);

const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(sum0123), _mm256_extractf128_ps(sum0123, 1));
const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));

sumf = _mm_cvtss_f32(r1);

// leftovers
for (int i = n32; i < n; ++i) {
	sumf += ggml_fp16_to_fp32(x[i])*ggml_fp16_to_fp32(y[i]);
}

#endif
*s = sumf;
}



inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * restrict y, ggml_fp16_t * restrict x, const float v) {

#ifdef __ARM_NEON
// NEON 128-bit

const int n64 = 64*(n/64);
const float16x8_t v8 = vdupq_n_f16(v);
float16x8_t x0, x1, x2, x3, x4, x5, x6, x7;
for (int i = 0; i < n64; i += 64) {
	y0 = vld1q_f16(y + i + 0 );
	y1 = vld1q_f16(y + i + 8 );
	y2 = vld1q_f16(y + i + 16);
	y3 = vld1q_f16(y + i + 24);
	y4 = vld1q_f16(y + i + 32);
	y5 = vld1q_f16(y + i + 40);
	y6 = vld1q_f16(y + i + 48);
	y7 = vld1q_f16(y + i + 56);

	x0 = vld1q_f16(x + i + 0 );
	x1 = vld1q_f16(x + i + 8 );
	x2 = vld1q_f16(x + i + 16);
	x3 = vld1q_f16(x + i + 24);
	x4 = vld1q_f16(x + i + 32);
	x5 = vld1q_f16(x + i + 40);
	x6 = vld1q_f16(x + i + 48);
	x7 = vld1q_f16(x + i + 56);

	y0 = vfmaq_f16(y0, x0, v8);
	y1 = vfmaq_f16(y1, x1, v8);
	y2 = vfmaq_f16(y2, x2, v8);
	y3 = vfmaq_f16(y3, x3, v8);
	y4 = vfmaq_f16(y4, x4, v8);
	y5 = vfmaq_f16(y5, x5, v8);
	y6 = vfmaq_f16(y6, x6, v8);
	y7 = vfmaq_f16(y7, x7, v8);

	vst1q_f16(y + i + 0 , y0);
	vst1q_f16(y + i + 8 , y1);
	vst1q_f16(y + i + 16, y2);
	vst1q_f16(y + i + 24, y3);
	vst1q_f16(y + i + 32, y4);
	vst1q_f16(y + i + 40 , y5);
	vst1q_f16(y + i + 48 , y6);
	vst1q_f16(y + i + 56 , y7);
}



// leftovers
for (int i = n64; i < n; ++i) {
	y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) + ggml_fp16_to_fp32(x[i])*v);
}

#else
// AVX 256-bit

const int n32 = 32*(n/32);
const __m256 v8 = _mm256_set1_ps(v);
__m256 x0, x1, x2, x3;
__m256 y0, y1, y2, y3;
for (int i = 0; i < n32; i += 32) {
	y0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 0 )));
	y1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 8 )));
	y2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 16)));
	y3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + i + 24)));

	x0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 0 )));
	x1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 8 )));
	x2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 16)));
	x3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + i + 24)));

	y0 = _mm256_fmadd_ps(x0, v8, y0);
	y1 = _mm256_fmadd_ps(x1, v8, y1);
	y2 = _mm256_fmadd_ps(x2, v8, y2);
	y3 = _mm256_fmadd_ps(x3, v8, y3);

	_mm_storeu_si128((__m128i*)(y + i + 0 ), _mm256_cvtps_ph(y0, 0));
	_mm_storeu_si128((__m128i*)(y + i + 8 ), _mm256_cvtps_ph(y1, 0));
	_mm_storeu_si128((__m128i*)(y + i + 16), _mm256_cvtps_ph(y2, 0));
	_mm_storeu_si128((__m128i*)(y + i + 24), _mm256_cvtps_ph(y3, 0));
}

for (int i = n32; i < n; ++i) {
	y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) + ggml_fp16_to_fp32(x[i])*v);
}
#endif
}

inline static void ggml_vec_scale_f32(const int n, float * y, const float v) { for (int i = 0; i < n; ++i) y[i] *= v; }
inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, x, x); *s = sqrt(*s); }
inline static void ggml_vec_sqr_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i]; }
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrt(x[i]); }
inline static void ggml_vec_abs_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void ggml_vec_sgn_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }

const ggml_float GELU_COEF_A = 0.044715;
const ggml_float SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

inline static void ggml_vec_gelu_f32 (const int n, float * y, const float * x) {
	for (int i = 0; i < n; ++i) {
	//y[i] = 0.5f*x[i]*(1.f + tanhf(SQRT_2_OVER_PI*(x[i] + 0.044715f*x[i]*x[i]*x[i])));
	//0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

		const ggml_float xx = x[i];
		y[i] = 0.5*xx*(1.0 + tanh(SQRT_2_OVER_PI*xx*(1.0 + GELU_COEF_A*xx*xx)));
	}
}

inline static void ggml_vec_sum_f32 (const int n, float * s, const float * x) { ggml_float sum = 0.0; for (int i = 0; i < n; ++i) sum += x[i]; *s += sum; }

inline static void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) { ggml_vec_norm_f32(n, s, x); *s = 1./(*s); }
```

</details>

<br/>

Этот код в библиотеке фактически выполняет все операции, все остальные операции по работе с матрицами и тензорами будут использовать код для работы с векторами в конечном счетё.

Вот к примеру, функция для вычисления абсолютного значения тензора, которая использует векторные операции для работы с данными.

##

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
        ggml_vec_abs_f32(nc, // <-- абсолютное значение вектора
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}
```

# Вычислительный граф

GGML использует вычислительный граф для организации операций над тензорами. Граф состоит из узлов, где каждый узел представляет собой операцию над тензорами. Узлы могут быть связаны друг с другом, образуя цепочку вычислений.

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

Граф содержит массивы узлов, градиентов и листьев, а также размер буфера для работы. Каждый узел в графе представляет собой тензор, который может быть результатом операции или входом для других операций.

Также заслуживает внимания функция для вычисления графа, которая запускает потоки для выполнения операций над тензорами (если нужно разбить на потоки). Здесь важно понимать что узлы графа топологически упорядочены, и каждый узел может зависеть от результатов предыдущих узлов. Поэтому, при вычислении графа, сначала выполняются узлы, от которых зависят другие узлы, а затем уже выполняются узлы, которые зависят от них.

<details>
<summary><b>Функция вычисления графа</b></summary>

```c
void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph) {
    if (cgraph->n_threads <= 0) {
        cgraph->n_threads = 8;
    }

    const int n_threads = cgraph->n_threads;

    struct ggml_compute_state_shared state_shared = {
        /*.spin      =*/ GGML_LOCK_INITIALIZER,
        /*.n_threads =*/ n_threads,
        /*.n_ready   =*/ 0,
        /*.has_work  =*/ false,
        /*.stop      =*/ false,
    };
    struct ggml_compute_state * workers = n_threads > 1 ? alloca(sizeof(struct ggml_compute_state)*(n_threads - 1)) : NULL;

    // create thread pool
    if (n_threads > 1) {
        ggml_lock_init(&state_shared.spin);

        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            workers[j] = (struct ggml_compute_state) {
                .thrd   = 0,
                .params = {
                    .type  = GGML_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = n_threads,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                },
                .node   = NULL,
                .shared = &state_shared,
            };
            int rc = pthread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
            assert(rc == 0);
            UNUSED(rc);
        }
    }

    // initialize tasks + work buffer
    {
        size_t work_size = 0;

        // thread scheduling for the different operations
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case GGML_OP_DUP:
                case GGML_OP_ADD:
                case GGML_OP_SUB:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                case GGML_OP_SQR:
                case GGML_OP_SQRT:
                case GGML_OP_SUM:
                case GGML_OP_MEAN:
                case GGML_OP_REPEAT:
                case GGML_OP_ABS:
                case GGML_OP_SGN:
                case GGML_OP_NEG:
                case GGML_OP_STEP:
                case GGML_OP_RELU:
                case GGML_OP_GELU:
                case GGML_OP_NORM:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        // TODO: use different scheduling for different matrix sizes
                        node->n_tasks = n_threads;

                        // TODO: better way to determine if the matrix is transposed
                        if (node->src0->nb[1] < node->src0->nb[0]) {
                            size_t cur = ggml_nbytes(node)*node->n_tasks; // TODO: this can become (n_tasks-1)
                            work_size = MAX(work_size, cur);
                        } else {
                            if (node->src0->type == GGML_TYPE_F16 &&
                                node->src1->type == GGML_TYPE_F32) {
                                size_t cur = sizeof(ggml_fp16_t)*ggml_nelements(node->src1);
                                work_size = MAX(work_size, cur);
                            }
                        }
                    } break;
                case GGML_OP_SCALE:
                case GGML_OP_CPY:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_GET_ROWS:
                case GGML_OP_DIAG_MASK_INF:
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_NONE:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_COUNT:
                    {
                        assert(false);
                    } break;
            };
        }

        if (cgraph->work != NULL && work_size > cgraph->work_size) {
            assert(false); // TODO: better handling
        }

        if (work_size > 0 && cgraph->work == NULL) {
            cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);

            GGML_PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
            cgraph->work = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cgraph->work_size);
        }
    }


    for (int i = 0; i < cgraph->n_nodes; i++) {

        struct ggml_tensor * node = cgraph->nodes[i];

        // INIT
        struct ggml_compute_params params = {
            /*.type  =*/ GGML_TASK_INIT,
            /*.ith   =*/ 0,
            /*.nth   =*/ n_threads,
            /*.wsize =*/ cgraph->work ? ggml_nbytes(cgraph->work) : 0,
            /*.wdata =*/ cgraph->work ? cgraph->work->data : NULL,
        };

        ggml_compute_forward(&params, node);

        // COMPUTE
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct ggml_compute_params) {
                    .type  = GGML_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = n_threads,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_store(&state_shared.has_work, true);
        }

        params.type = GGML_TASK_COMPUTE;
        ggml_compute_forward(&params, node);

        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }
        }

        // FINALIZE
        params.type = GGML_TASK_FINALIZE;
        ggml_compute_forward(&params, node);

    }

    // join thread pool
    if (n_threads > 1) {
        atomic_store(&state_shared.stop, true);
        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            int rc = pthread_join(workers[j].thrd, NULL);
            assert(rc == 0);
            UNUSED(rc);
        }

        ggml_lock_destroy(&state_shared.spin);
    }
}
```

</details>

<br/>

# Автоматическое дифференцирование и обучение

GGML поддерживает автоматическое дифференцирование и встроенные оптимизаторы, что позволяет обучать модели без ручного вычисления градиентов.

## Возможности GGML для обучения

- **Автоматическое дифференцирование**: GGML автоматически вычисляет градиенты
- **Встроенные оптимизаторы**: Adam, L-BFGS и другие алгоритмы оптимизации
- **Управление памятью**: Автоматическое управление графами вычислений
- **Обратное распространение**: Поддержка backward pass для градиентов

## Пример обучения линейной регрессии

Простой пример обучения линейной модели с помощью встроенного оптимизатора GGML:

<details>
<summary><b>Пример обучения линейной регрессии</b></summary>

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

**Особенности примера:**

- **Простота**: Линейная регрессия - одна из самых простых и надежных задач ML
- **Гарантированная сходимость**: Выпуклая задача оптимизации всегда сходится
- **Понятный результат**: Легко проверить правильность обучения
- **Минимальные требования**: Использует только базовые операции GGML

**Преимущества встроенного оптимизатора GGML:**

- Автоматическое управление памятью графов
- Оптимизированные алгоритмы (Adam, L-BFGS)
- Отсутствие проблем с переполнением стека
- Меньше кода и проще в использовании
