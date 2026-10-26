const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const start = source.indexOf('function thumbsModel(item)');
const end = source.indexOf('\nconst thumbsEnabled', start);
const context = { current: { settings: {} }, num: v => Number(v) || 0, svClamp: (v,a,b) => Math.min(b,Math.max(a,v)) };
vm.createContext(context);
vm.runInContext(source.slice(start,end), context);
const model = context.thumbsModel({ cover: 'data:image/jpeg;base64,AAAA', state: 'downloading', download_fraction: .1,
  thumbs: { duration: 120, frames: [{at:65, url:'frame.jpg', restored:false}] } });
assert.equal(model.frames.length,12);
assert.equal(model.frames[0].cover,true);
assert.equal(model.frames[0].restored,false);
assert.equal(model.frames[6].url,'frame.jpg');
assert.equal(model.frames[1].url,'');
assert.equal(model.dl,.1);
assert.equal(context.thumbsModel({state:'waiting'}),null);
assert.equal(context.thumbsModel({cover:'data:image/png;base64,AAAA', state:'waiting'}).frames.length,12);
context.current.settings.show_thumbnail_cover = false;
const withoutCover = context.thumbsModel({cover:'data:image/png;base64,AAAA', state:'done',
  thumbs:{duration:120, frames:[{at:5, url:'restored.jpg', restored:true}]}});
assert.equal(withoutCover.frames[0].url,'restored.jpg');
assert.equal(withoutCover.frames[0].restored,true);
assert.equal(withoutCover.frames[0].cover,undefined);
assert.equal(context.thumbsModel({cover:'data:image/png;base64,AAAA', state:'waiting'}),null);
console.log('Queue thumbnail model checks passed');
