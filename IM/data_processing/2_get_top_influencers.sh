#!/bin/bash
source_file="../retweetsranked_full.csv"
cut -d ";" -f 4 ${source_file} | sort | uniq -c | sed 's/[\ ]*//' | sort -rn -t " " -k 1,1 > 2_influencers.out
