# Where these pools come from

Plain wildcard `.txt` files can't carry comments (`WildcardTextFile.get_values()`
treats every non-blank line as a literal option, `#` included), so this file
holds the grounding that would otherwise live as Python comments. Kept as
`.md` rather than `.txt`/`.json`/`.yaml` so `WildcardManager`'s directory
scan never mistakes it for a wildcard collection.

Many of the citations below are Wikipedia rather than primary or more
authoritative art-historical literature -- adequate as a first pass, but
worth replacing with better sources later.

## medium.txt

Specific named 2D techniques only -- no photography, 3D/CGI rendering,
sculpture, or physical/textured craft media (embroidery, stained glass,
mosaic, tapestry). A picture "of" any of those collapses back into a photo
of a real object; only media where the artwork itself *is* the flat image
keep the LLM from defaulting to photorealism. Not drawn from an external
taxonomy -- just the working exclusion rule above, applied to a list of
common fine-art/print/drawing techniques.

## composition.txt

Two sources:
- The canonical **Elements and Principles of Design** taught in art
  education (balance, contrast, emphasis, movement, pattern, proportion,
  rhythm, unity) -- rule of thirds, golden spiral, radial/asymmetrical
  balance, and the contrast/emphasis/unity entries all come from here.
  See [Elements & Principles of Design (Tyler Museum of Art)](https://tylermuseum.art/2021/09/01/elements-and-principles-of-design/),
  [Composition (visual arts) -- Wikipedia](https://en.wikipedia.org/wiki/Composition_(visual_arts)).
- Named, documented compositional devices from art history: the
  [Golden triangle](https://en.wikipedia.org/wiki/Golden_triangle_(composition)),
  the [Rule of Odds](https://digital-photography-school.com/the-rule-of-odds-in-photography-an-easy-trick-for-better-compositions/),
  serpentine/S-curve composition (*figura serpentinata* --
  [Wikipedia](https://en.wikipedia.org/wiki/Figura_serpentinata)),
  [open vs. closed composition](https://photographylife.com/open-and-closed-composition),
  and the [tondo](https://en.wikipedia.org/wiki/Tondo_(art)) circular format.
- The one-point/two-point/isometric/atmospheric-perspective entries are
  standard perspective-drawing vocabulary, not attributed to a single source.

## subject.txt

Two classification systems:
- [Félibien's 1667 hierarchy of genres](https://en.wikipedia.org/wiki/Hierarchy_of_genres)
  (history/narrative painting, portraiture, genre scenes, landscape, still
  life, animal painting) -- covers most of the file.
- [Iconclass](https://en.wikipedia.org/wiki/Iconclass), the standard
  international classification system for iconographic subject matter, for
  what the genre hierarchy doesn't cover. Its 10 top-level divisions: 0
  Abstract/Non-representational, 1 Religion and Magic, 2 Nature, 3 Human
  being, 4 Society/Civilization/Culture, 5 Abstract Ideas and Concepts, 6
  History, 7 Bible, 8 Literature, 9 Classical Mythology and Ancient History.
  Divisions 1 and 7 (Religion and Magic, Bible) are deliberately not
  represented as their own entries -- religious content can still emerge
  through "mythological or folkloric" or "allegorical" if the LLM chooses
  it, just without a dedicated category steering toward it.

## palette.txt

Two sources:
- The seven standard **color-harmony schemes** from color theory
  (monochromatic, analogous, complementary, split-complementary, triadic,
  tetradic, achromatic).
- Real historical/period palettes, deliberately described by their actual
  pigments/qualities rather than by movement name (movement.txt already
  owns movement names -- naming a movement in both places risks it being
  named twice in one prompt from two independent axes): ukiyo-e's
  indigo-dominant palette with beni-red accents
  ([source](https://www.artelino.com/articles/aizuri-e.asp)), Fauvism's
  non-naturalistic color applied against local-color logic, and the Bauhaus
  primary triad (red/yellow/blue + black/white).

## mood.txt

Not drawn from an external taxonomy -- a deliberately distinct
emotional/atmospheric register, kept separate from palette (color alone
isn't mood; without this pool "mood" was a label with nothing actually
controlling it).

## movement.txt

Real, named, documented art-historical movements and traditions -- not
individual artists (models respond weakly and inconsistently to artist
names as an *axis value* fed into the synthesis call, and it reintroduces
tag-list/name-dropping bias; this is unrelated to whether the LLM's own
finished prose may name an artist, which the system prompt allows).
Broadly recognized collective styles with well-documented visual
conventions, spanning multiple eras and cultures, plus one contemporary
stylistic convention (graphic novel/comic-book illustration -- distinct
from "pen and ink drawing" in medium.txt, which is the raw technique, not
the style built on top of it).
