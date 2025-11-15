import fs from 'fs';
import csv from 'csv-parser';
import { QdrantClient } from '@qdrant/js-client-rest';
import { HfInference } from '@huggingface/inference';
import path from 'path';

const HF_MODEL = 'intfloat/multilingual-e5-base'; 
const FILE_PATH = path.join(__dirname, '../data/laws.csv');
const COLLECTION_NAME = process.env.QDRANT_COLLECTION || 'nitisure_laws';

if (!process.env.HF_TOKEN || !process.env.QDRANT_URL || !process.env.QDRANT_KEY) {
  console.error('❌ MISSING KEYS: GitHub Secrets are not set!');
  process.exit(1);
}

const hf = new HfInference(process.env.HF_TOKEN);
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL, apiKey: process.env.QDRANT_KEY });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function main() {
  console.log('🚀 Starting Ingestion...');

  try {
    const result = await qdrant.getCollections();
    const exists = result.collections.some((c) => c.name === COLLECTION_NAME);
    if (!exists) {
      console.log(`📦 Creating collection: ${COLLECTION_NAME}`);
      await qdrant.createCollection(COLLECTION_NAME, { vectors: { size: 768, distance: 'Cosine' } });
    }
  } catch (err) {
    console.error('❌ Qdrant Connection Error:', err);
    process.exit(1);
  }

  const rows: any[] = [];
  // Safe check for file existence
  if (!fs.existsSync(FILE_PATH)) {
     console.error(`❌ File not found at: ${FILE_PATH}`);
     process.exit(1);
  }

  fs.createReadStream(FILE_PATH)
    .pipe(csv())
    .on('data', (data) => rows.push(data))
    .on('end', async () => {
      console.log(`📊 Found ${rows.length} rows.`);

      for (const row of rows) {
        if (!row.text_th) continue;

        console.log(`🔹 Processing: ${row.section_number_eng}`);

        const textToEmbed = `
          Law: ${row.act_name_thai} (${row.act_name_eng})
          Section: ${row.section_number_eng}
          Thai Text: ${row.text_th}
          Eng Text: ${row.text_eng}
          Explanation: ${row.notes_thai}
          Keywords: ${row.keywords_th}, ${row.keywords_eng}
          Cases: ${row.related_cases}
        `.trim();

        try {
          const embedding = await hf.featureExtraction({
            model: HF_MODEL,
            inputs: textToEmbed,
          });

          await qdrant.upsert(COLLECTION_NAME, {
            points: [{
              id: crypto.randomUUID(),
              // FIX IS HERE: We force it to be a number array
              vector: embedding as unknown as number[], 
              payload: {
                act_name: row.act_name_thai,
                section: row.section_number_eng,
                text: row.text_th,
                explanation: row.notes_thai,
                url: row.source_url,
                category: row.law_category
              }
            }]
          });
          console.log(`✅ Uploaded ${row.section_number_eng}`);
          await sleep(300);
        } catch (error) {
          console.error(`❌ Failed Row:`, error);
        }
      }
      console.log('🎉 INGESTION COMPLETE!');
    });
}

main();
