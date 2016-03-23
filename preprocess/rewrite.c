//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries
const long long max_word = 102100;       // max number of words in dictionary

int main(int argc, char **argv) {
  FILE *f;
  FILE *g;
  FILE *h;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size], word_list[max_size], word_emb[max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, e, i, cn, bi[100];
  char ch;
  float *M;
  char *vocab;
  char word[max_word][max_w];
  
  if (argc < 4) {
    printf("Usage: ./distance <FILE> <WORD_LIST> <WORD_EMB>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  strcpy(word_list, argv[2]);
  strcpy(word_emb, argv[3]);

  // Load dictionary.
  g = fopen(word_list, "rb");
  char line[max_w];
  e = 0;
  while(fgets(line, sizeof(line), g)){
    char * w = strtok(line, "\n");
    if (w == NULL) continue;
    strcpy(word[e], w);
    /* printf("%s\n", word[e]); */
    e++;
  }
  
  fclose(g);

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  h = fopen(word_emb, "w");
  // for each word in the dictionary, output two lines
  // word
  // <embedding> seperated by ' '
  for (a = 0; a < max_word; a++){
    // For each word search the pretrained emb.
    for (b = 0; b < words; b++){
      if (!strcmp(word[a], &vocab[b*max_w])) break;
    }
    if (b != words){
      // Found
      if (a % 1000 == 0){
	printf("Found %lld\n", a);
      }
      fprintf(h, "%s\n", word[a]);
      for (c = 0; c < size-1; c++){
	// Output each embedding element except the last one.
	fprintf(h, "%f ", M[c + b * size]);
      }
      // Output the last embedding element.
      fprintf(h, "%f\n", M[b * size + size - 1]);
    }
  }
  fclose(h);
  
  return 0;
}
