// module
#import "@preview/chic-hdr:0.5.0":*
#import "@preview/equate:0.3.2":*

// constraint
#import "theme.typ": *

// page + text + paragraph
#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: body_font,
  size: 11.5pt,
  fill: body_color,
  lang: "vi",
)

#set par(
  justify: true,
  first-line-indent: (amount: 1.2em, all: true),
  leading: 0.72em,
)

#set list(indent: 1.5em)

// numbering
#set heading(
  numbering: "1.1",
  supplement: [Mục],
)

// = phần chính
#show heading.where(level: 1): it => {
  let nums = counter(heading).at(it.location())
  let s = nums.map(str).join(".")
  pagebreak(weak: true)
  v(0.2em)
  if it.numbering != none {
    let nums = counter(heading).at(it.location())
    let s = nums.map(str).join(".")
    text(fill: title_color, font: body_font, size: 20pt)[
      #s #h(0.1em)
      #box(width: 1.2pt, height: 1.1em, fill: title_color.darken(30%), baseline: 20%)
      #h(0.2em)
      #it.body
    ]
  } else {
    text(fill: title_color, font: body_font, size: 20pt)[
      #it.body
    ]
  }
  v(0.8em)
}

// == phụ lục cấp 2
#show heading.where(level: 2): it => [
  #text(font: body_font, size: 16pt, fill: title_color,)[#it]
]

// === phụ lục cấp 3
#show heading.where(level: 3): it => [
  #text(font: body_font, fill: title_color,)[#it] #v(0.5em)
]
#show heading.where(level: 3): set heading(outlined: false)


// non-numbering
#show selector(<nonumber>): set heading(
  numbering: none
)

// non-outlined
#show selector(<nooutlined>): set heading(
  outlined: false,
)

// math
#set math.equation(numbering: "(1.1)")
#show: equate.with(breakable: true, sub-numbering: true)

// figure caption
#show figure.caption: it => [
  #set text(size: 10pt, fill: soft_color)
  #strong[#it.supplement #it.counter.display(it.numbering)]
  #it.separator
  #it.body
]

#show figure.where(kind: table): set figure.caption(position: top)

// table
#show table.cell: set par(
  justify: false,
  first-line-indent: 0em,
)

//// end of config
#include "chapters/cover.typ"
#pagebreak()

// header/footer
#let hf-style(body) = text(fill: rgb("#677CA6"), size: 10pt, body)



// Mục lục
#counter(page).update(1)
#show: chic.with(
  chic-header(
    left-side: hf-style[*FIT-HCMUS*],
    right-side: hf-style[*University of Science - VNUHCM*],
  ),
  chic-footer(
    left-side: hf-style[*Toán ứng dụng và thống kê*],
    right-side: hf-style(strong(context [#counter(page).display() / #counter(page).final().at(0)])),
  ),
  chic-separator(0.6pt + rgb("#A5B4D6")),
  chic-offset(14pt),
)

#show outline.entry.where(level: 1): it => {
  v(12pt, weak: true)
  strong(text(fill: title_color, it))
}

#outline(title: [Mục lục], depth: 3, indent: auto)

#include "chapters/group_work.typ"

#include "chapters/intro.typ"
#include "chapters/part1_gauss.typ"
#include "chapters/part2_decomposition.typ"
#include "chapters/part3_solve_analysis.typ"
#include "chapters/conclude.typ"
#include "chapters/appendix.typ"
#include "chapters/references.typ"
