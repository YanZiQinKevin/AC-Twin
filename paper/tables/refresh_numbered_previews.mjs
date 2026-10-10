import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const root=process.cwd();
for(const [num,name,range] of [[6,'Table_6_Confirmed_Results.xlsx','A1:G18'],[7,'Table_7_Published_Comparison.xlsx','A1:I18'],[8,'Table_8_Supervised_Controls.xlsx','A1:C18']]) {
 const file=path.join(root,`paper/tables/table${num}`,name);
 const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
 const sheet=wb.worksheets.getItemAt(0);
 const img=await wb.render({sheetName:sheet.name,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(root,`paper/tables/_preview_main_tables/Table_${num}.png`),new Uint8Array(await img.arrayBuffer()));
 console.log(`Table ${num} saved-file preview refreshed`);
}
