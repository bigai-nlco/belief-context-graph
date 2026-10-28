import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const css = name => readFileSync(new URL('../'+name,import.meta.url),'utf8');
const variables = text => Object.fromEntries([...text.matchAll(/(--[\w-]+)\s*:\s*(#[\da-f]{6})\b/gi)].map(m=>[m[1],m[2].toLowerCase()]));
const contrast = (a,b) => {
  const luminance = hex => {
    const rgb=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);
    return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722;
  };
  const x=luminance(a),y=luminance(b);
  return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);
};

test('workbench and copy editor share the ARR navy, burgundy and neutral palette', () => {
  const expected = {'--paper':'#ffffff','--ink':'#182330','--muted':'#4a5663','--navy':'#162638','--navy-light':'#243b52','--red':'#9f343c','--red-dark':'#78262d','--surface':'#f4f6f7','--surface-strong':'#e9edef','--line':'#d8dee3','--line-strong':'#b7c0c8','--focus':'#2b6cb0'};
  for(const name of ['styles.css','copy-editor.css']){
    const text=css(name),theme=variables(text);
    for(const [key,color] of Object.entries(expected))assert.equal(theme[key],color,`${name}: ${key}`);
    const defined=new Set([...text.matchAll(/(--[\w-]+)\s*:/g)].map(m=>m[1]));
    for(const match of text.matchAll(/var\((--[\w-]+)/g))assert(defined.has(match[1]),`${name}: missing ${match[1]}`);
    assert.doesNotMatch(text,/var\(--(?:green|green-dark|lime|soft)\)/,'no old green theme references');
  }
});

test('primary, secondary, selected and warning text remain readable', () => {
  for(const name of ['styles.css','copy-editor.css']){
    const theme=variables(css(name));
    for(const [foreground,background] of [
      ['--ink','--paper'],['--muted','--surface'],['--muted','--surface-strong'],
      ['--paper','--navy'],['--paper','--navy-light'],['--red-dark','--red-soft'],
      ['--paper','--red'],['--warning','--warning-bg'],
    ])assert(contrast(theme[foreground],theme[background])>=4.5,`${name}: ${foreground} on ${background}`);
  }
});

test('graph, logo and copy selection use the new theme without remote assets', () => {
  assert.match(css('styles.css'),/\.svg-node\.center circle\{fill:var\(--navy\);stroke:var\(--navy\)/);
  assert.match(css('app.mjs'),/fill="var\(--line-strong\)"/);
  assert.match(css('icon.svg'),/fill="#162638"/);
  assert.match(css('copy-runtime.css'),/#9f343c/);
  for(const name of ['styles.css','copy-editor.css','copy-runtime.css'])assert.doesNotMatch(css(name),/@import|url\(\s*["']?https?:/i);
});
