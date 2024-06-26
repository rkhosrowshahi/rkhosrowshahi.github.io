"use strict";(self.webpackChunkrkhosrowshahi_github_io=self.webpackChunkrkhosrowshahi_github_io||[]).push([[986,660,672],{7660:(e,a,s)=>{s.r(a),s.d(a,{default:()=>r});s(5043),s(4050);var t=s(579);const r=function(e){const{title:a}=e;return(0,t.jsx)("div",{className:"header",children:a})}},9986:(e,a,s)=>{s.r(a),s.d(a,{default:()=>h});var t=s(5043),r=s(3519),c=s(1072),o=s(4282),n=s(4574),l=s(8885),d=s.n(l),i=s(7660),f=s(5724),m=s(7672),x=s(4331),u=s(579);const b={containerStyle:{marginBottom:25},showMoreStyle:{margin:25}},h=e=>{var a;const s=(0,t.useContext)(n.Dx),{header:l}=e,[h,y]=(0,t.useState)(null),[p,g]=(0,t.useState)(!1);(0,t.useEffect)((()=>{fetch(f.A.publications,{method:"GET"}).then((e=>e.json())).then((e=>{g(e.publications.length>=3),y(e)})).catch((e=>e))}),[]);const N=p&&h?h.publications.length:3;return(0,u.jsxs)(u.Fragment,{children:[(0,u.jsx)(i.default,{title:l}),h?(0,u.jsx)("div",{className:"section-content-container",children:(0,u.jsxs)(r.A,{style:b.containerStyle,children:[(0,u.jsx)(c.A,{xs:1,sm:1,md:2,lg:3,className:"g-4",children:null===(a=h.publications)||void 0===a?void 0:a.slice(0,N).map((e=>(0,u.jsx)(d(),{children:(0,u.jsx)(m.default,{publications:e})},e.title)))}),!p&&(0,u.jsx)(o.A,{style:b.showMoreStyle,variant:s.bsSecondaryVariant,onClick:()=>g(!0),children:"show more"})]})}):(0,u.jsx)(x.default,{})]})}},7672:(e,a,s)=>{s.r(a),s.d(a,{default:()=>H});var t=s(5043),r=s(8602),c=s(8139),o=s.n(c),n=s(7852),l=s(579);const d=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r="div",...c}=e;return t=(0,n.oU)(t,"card-body"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));d.displayName="CardBody";const i=d,f=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r="div",...c}=e;return t=(0,n.oU)(t,"card-footer"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));f.displayName="CardFooter";const m=f;var x=s(1778);const u=t.forwardRef(((e,a)=>{let{bsPrefix:s,className:r,as:c="div",...d}=e;const i=(0,n.oU)(s,"card-header"),f=(0,t.useMemo)((()=>({cardHeaderBsPrefix:i})),[i]);return(0,l.jsx)(x.A.Provider,{value:f,children:(0,l.jsx)(c,{ref:a,...d,className:o()(r,i)})})}));u.displayName="CardHeader";const b=u,h=t.forwardRef(((e,a)=>{let{bsPrefix:s,className:t,variant:r,as:c="img",...d}=e;const i=(0,n.oU)(s,"card-img");return(0,l.jsx)(c,{ref:a,className:o()(r?"".concat(i,"-").concat(r):i,t),...d})}));h.displayName="CardImg";const y=h,p=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r="div",...c}=e;return t=(0,n.oU)(t,"card-img-overlay"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));p.displayName="CardImgOverlay";const g=p,N=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r="a",...c}=e;return t=(0,n.oU)(t,"card-link"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));N.displayName="CardLink";const j=N;var v=s(4488);const w=(0,v.A)("h6"),S=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r=w,...c}=e;return t=(0,n.oU)(t,"card-subtitle"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));S.displayName="CardSubtitle";const C=S,P=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r="p",...c}=e;return t=(0,n.oU)(t,"card-text"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));P.displayName="CardText";const k=P,R=(0,v.A)("h5"),A=t.forwardRef(((e,a)=>{let{className:s,bsPrefix:t,as:r=R,...c}=e;return t=(0,n.oU)(t,"card-title"),(0,l.jsx)(r,{ref:a,className:o()(s,t),...c})}));A.displayName="CardTitle";const U=A,T=t.forwardRef(((e,a)=>{let{bsPrefix:s,className:t,bg:r,text:c,border:d,body:f=!1,children:m,as:x="div",...u}=e;const b=(0,n.oU)(s,"card");return(0,l.jsx)(x,{ref:a,...u,className:o()(t,b,r&&"bg-".concat(r),c&&"text-".concat(c),d&&"border-".concat(d)),children:f?(0,l.jsx)(i,{children:m}):m})}));T.displayName="Card";const B=Object.assign(T,{Img:y,Title:U,Subtitle:C,Body:i,Link:j,Text:k,Header:b,Footer:m,ImgOverlay:g});var F=s(4282);const I=t.forwardRef(((e,a)=>{let{bsPrefix:s,bg:t="primary",pill:r=!1,text:c,className:d,as:i="span",...f}=e;const m=(0,n.oU)(s,"badge");return(0,l.jsx)(i,{ref:a,...f,className:o()(d,m,r&&"rounded-pill",c&&"text-".concat(c),t&&"bg-".concat(t))})}));I.displayName="Badge";const V=I;var _=s(4574),E=s(4819);const D={badgeStyle:{paddingLeft:10,paddingRight:10,paddingTop:5,paddingBottom:5,margin:5},cardStyle:{borderRadius:10},cardTitleStyle:{fontSize:24,fontWeight:700},cardTextStyle:{textAlign:"left"},linkStyle:{textDecoration:"none",padding:10},buttonStyle:{margin:5}},H=e=>{var a;const s=(0,t.useContext)(_.Dx),{publications:c}=e;return(0,l.jsx)(r.A,{children:(0,l.jsxs)(B,{style:{...D.cardStyle,backgroundColor:s.cardBackground,borderColor:s.cardBorderColor},text:s.bsSecondaryVariant,children:[(0,l.jsx)(B.Img,{variant:"top",src:null===c||void 0===c?void 0:c.image}),(0,l.jsxs)(B.Body,{children:[(0,l.jsx)(B.Title,{style:D.cardTitleStyle,children:c.title}),(0,l.jsx)(B.Text,{style:D.cardTextStyle,children:(o=c.bodyText,(0,l.jsx)(E.$,{children:o}))})]}),(0,l.jsx)(B.Body,{children:null===c||void 0===c||null===(a=c.links)||void 0===a?void 0:a.map((e=>(0,l.jsx)(F.A,{style:D.buttonStyle,variant:"outline-"+s.bsSecondaryVariant,onClick:()=>window.open(e.href,"_blank"),children:e.text},e.href)))}),c.tags&&(0,l.jsx)(B.Footer,{style:{backgroundColor:s.cardFooterBackground},children:c.tags.map((e=>(0,l.jsx)(V,{pill:!0,bg:s.bsSecondaryVariant,text:s.bsPrimaryVariant,style:D.badgeStyle,children:e},e)))})]})});var o}},4282:(e,a,s)=>{s.d(a,{A:()=>i});var t=s(8139),r=s.n(t),c=s(5043),o=s(4140),n=s(7852),l=s(579);const d=c.forwardRef(((e,a)=>{let{as:s,bsPrefix:t,variant:c="primary",size:d,active:i=!1,disabled:f=!1,className:m,...x}=e;const u=(0,n.oU)(t,"btn"),[b,{tagName:h}]=(0,o.Am)({tagName:s,disabled:f,...x}),y=h;return(0,l.jsx)(y,{...b,...x,ref:a,disabled:f,className:r()(m,u,i&&"active",c&&"".concat(u,"-").concat(c),d&&"".concat(u,"-").concat(d),x.href&&f&&"disabled")})}));d.displayName="Button";const i=d},8602:(e,a,s)=>{s.d(a,{A:()=>d});var t=s(8139),r=s.n(t),c=s(5043),o=s(7852),n=s(579);const l=c.forwardRef(((e,a)=>{const[{className:s,...t},{as:c="div",bsPrefix:l,spans:d}]=function(e){let{as:a,bsPrefix:s,className:t,...c}=e;s=(0,o.oU)(s,"col");const n=(0,o.gy)(),l=(0,o.Jm)(),d=[],i=[];return n.forEach((e=>{const a=c[e];let t,r,o;delete c[e],"object"===typeof a&&null!=a?({span:t,offset:r,order:o}=a):t=a;const n=e!==l?"-".concat(e):"";t&&d.push(!0===t?"".concat(s).concat(n):"".concat(s).concat(n,"-").concat(t)),null!=o&&i.push("order".concat(n,"-").concat(o)),null!=r&&i.push("offset".concat(n,"-").concat(r))})),[{...c,className:r()(t,...d,...i)},{as:a,bsPrefix:s,spans:d}]}(e);return(0,n.jsx)(c,{...t,ref:a,className:r()(s,!d.length&&l)})}));l.displayName="Col";const d=l},1072:(e,a,s)=>{s.d(a,{A:()=>d});var t=s(8139),r=s.n(t),c=s(5043),o=s(7852),n=s(579);const l=c.forwardRef(((e,a)=>{let{bsPrefix:s,className:t,as:c="div",...l}=e;const d=(0,o.oU)(s,"row"),i=(0,o.gy)(),f=(0,o.Jm)(),m="".concat(d,"-cols"),x=[];return i.forEach((e=>{const a=l[e];let s;delete l[e],null!=a&&"object"===typeof a?({cols:s}=a):s=a;const t=e!==f?"-".concat(e):"";null!=s&&x.push("".concat(m).concat(t,"-").concat(s))})),(0,n.jsx)(c,{ref:a,...l,className:r()(t,d,...x)})}));l.displayName="Row";const d=l}}]);
//# sourceMappingURL=986.520a59a8.chunk.js.map