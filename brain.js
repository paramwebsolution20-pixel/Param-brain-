'use strict';
// PARAM BRAIN V1: tiny single-head causal Transformer trained from scratch.
const $ = id => document.getElementById(id);
const CONTEXT=12, DIM=16, HIDDEN=32, RATE=.008;
let chars=[],toId=new Map(),weights={},optimizer=null,training=false,stopRequested=false,trained=0,corpusIds=[];
const names=['token','position','wq','wk','wv','wo','w1','b1','w2','b2','head','bias'];
function createWeights(vocab){
 const v=(shape,scale=.08)=>tf.variable(tf.randomNormal(shape,0,scale));
 return {token:v([vocab,DIM]),position:v([CONTEXT,DIM]),wq:v([DIM,DIM]),wk:v([DIM,DIM]),wv:v([DIM,DIM]),wo:v([DIM,DIM]),w1:v([DIM,HIDDEN]),b1:v([HIDDEN],0),w2:v([HIDDEN,DIM]),b2:v([DIM],0),head:v([DIM,vocab]),bias:v([vocab],0)};
}
function disposeModel(){Object.values(weights).forEach(w=>w.dispose());weights={};if(optimizer){optimizer.dispose();optimizer=null;}}
function model(ids){
 // ids shape [1, CONTEXT]. One causal self-attention head.
 const positions=tf.range(0,CONTEXT,1,'int32');
 const x=tf.add(tf.gather(weights.token,ids),tf.expandDims(tf.gather(weights.position,positions),0));
 const q=tf.matMul(x,weights.wq),k=tf.matMul(x,weights.wk),v=tf.matMul(x,weights.wv);
 const scores=tf.div(tf.matMul(q,k,false,true),Math.sqrt(DIM));
 const mask=tf.tensor2d(Array.from({length:CONTEXT*CONTEXT},(_,i)=>Math.floor(i/CONTEXT)>=i%CONTEXT?0:-1e9),[CONTEXT,CONTEXT]);
 const att=tf.matMul(tf.softmax(tf.add(scores,mask),-1),v);
 const residual=tf.add(x,tf.matMul(att,weights.wo));
 const hidden=tf.relu(tf.add(tf.matMul(residual,weights.w1),weights.b1));
 const y=tf.add(residual,tf.add(tf.matMul(hidden,weights.w2),weights.b2));
 return tf.add(tf.matMul(y,weights.head),weights.bias);
}
function encode(text){return [...text].map(c=>toId.has(c)?toId.get(c):0);}
function setup(text){
 if([...text].length<CONTEXT+2)throw Error('Add more training text.');
 disposeModel();chars=[...new Set([...text])].sort();toId=new Map(chars.map((c,i)=>[c,i]));corpusIds=encode(text);
 weights=createWeights(chars.length);optimizer=tf.train.adam(RATE);trained=0;
 $('modelStats').textContent=`${chars.length} characters · context ${CONTEXT} · ${Object.values(weights).reduce((s,w)=>s+w.size,0).toLocaleString()} parameters`;
 $('train').disabled=false;$('generate').disabled=false;$('save').disabled=false;$('status').textContent='Model created';$('loss').textContent='—';$('progress').style.width='0%';
}
function oneStep(){
 const start=Math.floor(Math.random()*(corpusIds.length-CONTEXT));
 const input=tf.tensor2d([corpusIds.slice(start,start+CONTEXT)],[1,CONTEXT],'int32');
 const target=tf.tensor1d(corpusIds.slice(start+1,start+CONTEXT+1),'int32');
 const cost=optimizer.minimize(()=>tf.tidy(()=>{
   const logits=model(input).reshape([CONTEXT,chars.length]);
   const labels=tf.oneHot(target,chars.length);
   return tf.neg(tf.mean(tf.sum(tf.mul(labels,tf.logSoftmax(logits,-1)),1)));
 }),true,Object.values(weights));
 const value=cost.dataSync()[0];cost.dispose();input.dispose();target.dispose();return value;
}
async function train(){
 if(training)return;training=true;stopRequested=false;$('train').disabled=true;$('init').disabled=true;$('stop').disabled=false;$('generate').disabled=true;$('save').disabled=true;
 try{for(let i=0;i<50&&!stopRequested;i++){
   const loss=oneStep();trained++;$('loss').textContent=loss.toFixed(4);$('status').textContent=`${trained} steps`; $('progress').style.width=`${(i+1)*2}%`;
   if(i%2===0)await tf.nextFrame();
 }}catch(e){alert('Training error: '+e.message);}finally{training=false;$('train').disabled=false;$('init').disabled=false;$('stop').disabled=true;$('generate').disabled=false;$('save').disabled=false;}
}
async function generate(){
 $('generate').disabled=true;
 try{
  let output=$('prompt').value||chars[0];let sequence=encode(output);
  for(let i=0;i<80;i++){
   const window=sequence.slice(-CONTEXT);while(window.length<CONTEXT)window.unshift(0);
   const probs=tf.tidy(()=>{
    const ids=tf.tensor2d([window],[1,CONTEXT],'int32');
    const logits=model(ids).slice([0,CONTEXT-1,0],[1,1,chars.length]).reshape([chars.length]);
    return tf.softmax(tf.div(logits,0.8));
   });
   const values=await probs.data();probs.dispose();let r=Math.random(),chosen=values.length-1;
   for(let j=0;j<values.length;j++){r-=values[j];if(r<=0){chosen=j;break;}}
   sequence.push(chosen);output+=chars[chosen];
   if(i%10===0)await tf.nextFrame();
  }
  $('output').textContent=output;
 }catch(e){alert('Generation error: '+e.message);}finally{$('generate').disabled=false;}
}
function exportWeights(){
 const data={version:1,context:CONTEXT,dim:DIM,chars,trained,weights:{}};
 for(const name of names)data.weights[name]={shape:weights[name].shape,values:Array.from(weights[name].dataSync())};
 const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='param-brain-v1-weights.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);
}
async function restore(file){
 if(!file)return;
 try{
  const data=JSON.parse(await file.text());
  if(data.version!==1||data.context!==CONTEXT||data.dim!==DIM||!Array.isArray(data.chars)||!data.chars.length||!data.weights)throw Error('Incompatible model file');
  const text=$('corpus').value;setup(text);
  if(JSON.stringify(chars)!==JSON.stringify(data.chars))throw Error('Training text character set differs from saved model. Restore original training text first.');
  for(const name of names){const item=data.weights[name];if(!item||JSON.stringify(item.shape)!==JSON.stringify(weights[name].shape)||item.values.length!==weights[name].size)throw Error('Invalid weight: '+name);weights[name].assign(tf.tensor(item.values,item.shape));}
  trained=data.trained||0;$('status').textContent=`Restored · ${trained} steps`;
 }catch(e){alert('Import failed: '+e.message);}
}
$('init').onclick=()=>{try{setup($('corpus').value);}catch(e){alert(e.message);}};
$('train').onclick=train;$('stop').onclick=()=>{stopRequested=true;};$('generate').onclick=generate;$('save').onclick=exportWeights;
$('restore').onchange=e=>restore(e.target.files[0]);
if(typeof tf==='undefined'){$('status').textContent='TensorFlow.js could not load. Connect to the internet and reload.';}else{$('status').textContent='TensorFlow.js loaded · '+tf.getBackend();}
