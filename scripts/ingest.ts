// @ts-nocheck
import fs from 'fs';
import csv from 'csv-parser';
import { QdrantClient } from '@qdrant/js-client-rest';
import { HfInference } from '@huggingface/inference';
import path from 'path';

// --- CONFIGURATION ---
const HF_MODEL = 'intfloat/multilingual-e5-base'; 
const COLLECTION_NAME = process.env.QDRANT_COLLECTION || 'nitisure_laws';

// SAFE PATH FINDER: This works better on GitHub servers
const FILE_PATH = path.join(process.cwd(), 'data', 'laws.csv');

// --- CHECK KEYS ---
if (!process.env.HF_TOKEN || !process.env.QDRANT_URL || !process.env.QDRANT_KEY) {
  console.error('❌ CRITICAL ERROR: Missing API Keys!');
  console.error('Please check GitHub Settings > Secrets and variables > Actions');
  process.exit(1);
}

const hf = new HfInference(process.env.HF_TOKEN);
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL, apiKey: process.env.QDRANT_KEY });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  console.log('🚀 Starting Ingestion (Nuclear Mode)...');
  console.log(`📂 Looking for file at: ${FILE_PATH}`);

  // 1. DEBUG: Check if file exists
  if (!fs.existsSync(FILE_PATH)) {
    console.error('❌ FILE NOT FOUND!');
    console.error('Current Folder contents:', fs.readdirSync(process.cwd()));
    console.error('Data Folder contents:', fs.existsSync('data') ? fs.readdirSync('data') : 'No data folder');
    process.exit(1);
  }

  // 2. SETUP DATABASE
  try {
    const result = await qdrant.getCollections();
    const exists = result.collections.some((c) => c.name === COLLECTION_NAME);
    if (!exists) {
      console.log(`📦 Creating collection: ${COLLECTION_NAME}`);
      await qdrant.createCollection(COLLECTION_NAME, { vectors: { size: 768, distance: 'Cosine' } });
    }
  } catch (err) {
    console.error('❌ QDRANT CONNECTION FAILED');
    console.error('Check your QDRANT_URL and QDRANT_KEY.');
    console.error('Error details:', err.message);
    process.exit(1);
  }

  // 3. READ CSV
  const rows = [];
  fs.createReadStream(FILE_PATH)
    .pipe(csv())
    .on('data', (data) => rows.push(data))
    .on('end', async () => {
      console.log(`📊 Found ${rows.length} rows.`);

      for (const row of rows) {
        if (!row.text_th) continue;

        console.log(`🔹 Processing: ${row.section_number_eng}`);

        // COMBINE DATA
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
          // GENERATE VECTOR
          const embedding = await hf.featureExtraction({
            model: HF_MODEL,
            inputs: textToEmbed,
          });

          // UPLOAD TO QDRANT
          await qdrant.upsert(COLLECTION_NAME, {
            points: [{
              id: crypto.randomUUID(),
              vector: embedding, // No type check needed due to @ts-nocheck
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
          await sleep(500);
        } catch (error) {
          console.error(`❌ Failed Row:`, error.message);
        }
      }
      console.log('🎉 INGESTION COMPLETE!');
    });
}

main();
