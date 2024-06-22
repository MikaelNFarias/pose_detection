rm data/output/train/annotations/render/*
rm data/output/train/annotations/measurements/*
rm data/output/train/annotations/plane/*

rm data/output/train/frontal/*
rm data/output/train/side/*
truncate -s 0 data/schemas/train/train_schema.json
truncate -s 0 data/schemas/train/train_schema.json