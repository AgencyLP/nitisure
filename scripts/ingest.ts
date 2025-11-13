import fs from 'fs';
import csv from 'csv-parser';
import { QdrantClient } from '@qdrant/js-client-rest';
import { HfInference } from '@huggingface/inference';
import path from 'path';

const HF_MODEL = 'intfloat/multilingual-e5-base'; 
// This path works for GitHub Actions
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
  console.log('🚀 Starting Ingestion (Custom Data Structure)...');

  // 1. Setup Database
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

  // 2. Read CSV
  const rows: any[] = [];
  fs.createReadStream(FILE_PATH)
    .pipe(csv())
    .on('data', (data) => rows.push(data))
    .on('end', async () => {
      console.log(`📊 Found ${rows.length} rows.`);

      for (const row of rows) {
        // Skip if no Thai text (empty row)
        if (!row.text_th) continue;

        console.log(`🔹 Processing: ${row.section_number_eng}`);

        // --- THE BRAIN: What the AI "Reads" to find the law ---
        // We combine English, Thai, Notes, and Cases into one searchable block
        const textToEmbed = `
          Law: ${row.act_name_thai} (${row.act_name_eng})
          Section: ${row.section_number_eng} / ${row.section_number_thai}
          Thai Text: ${row.text_th}
          English Text: ${row.text_eng}
          Thai Explanation: ${row.notes_thai}
          English Explanation: ${row.notes_eng}
          Keywords: ${row.keywords_th}, ${row.keywords_eng}
          Relevant Supreme Court Case: ${row.related_cases}
        `.trim();

        try {
          // Generate Vector
          const embedding = await hf.featureExtraction({
            model: HF_MODEL,
            inputs: textToEmbed,
          });

          // --- THE DISPLAY: What the User "Sees" ---
          // We upload the clean data to show on the website
          await qdrant.upsert(COLLECTION_NAME, {
            points: [{
              id: crypto.randomUUID(),
              vector: embedding,
              payload: {
                act_name: row.act_name_thai,
                section: row.section_number_eng, // "Section 295"
                text: row.text_th,               // Thai Law Text
                text_eng: row.text_eng,          // English Law Text (Bonus!)
                explanation: row.notes_thai,     // Simplified Thai
                category: row.law_category,
                url: row.source_url
              }
            }]
          });
          console.log(`✅ Uploaded ${row.section_number_eng}`);
          await sleep(300); // Rest to avoid hitting limits
        } catch (error) {
          console.error(`❌ Failed Row ${row.section_number_eng}:`, error);
        }
      }
      console.log('🎉 GITHUB INGESTION COMPLETE!');
    });
}

main();
