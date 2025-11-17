# Adding Investing Legends Images

The Hero section now includes subtle background images honoring investing legends. Currently, it uses placeholder images from Unsplash. To add actual photos of your idols, follow these steps:

## Quick Setup

1. **Collect Images**: Gather professional photos of each legend (preferably speaking/presenting photos)
   - Warren Buffett
   - Charlie Munger
   - Stan Druckenmiller
   - Ben Graham
   - John Maynard Keynes
   - David Tepper
   - Peter Lynch
   - Terry Smith

2. **Save Images**: Place them in `caria_data/caria-app/public/images/legends/`
   ```
   public/
     images/
       legends/
         warren-buffett.jpg
         charlie-munger.jpg
         stan-druckenmiller.jpg
         ben-graham.jpg
         john-keynes.jpg
         david-tepper.jpg
         peter-lynch.jpg
         terry-smith.jpg
   ```

3. **Update Hero Component**: Edit `components/Hero.tsx` and replace the Unsplash URLs:

```tsx
{/* Warren Buffett */}
<div className="legend-figure legend-buffett"
     style={{backgroundImage: "url('/images/legends/warren-buffett.jpg')"}}></div>

{/* Charlie Munger */}
<div className="legend-figure legend-munger"
     style={{backgroundImage: "url('/images/legends/charlie-munger.jpg')"}}></div>

{/* Stan Druckenmiller */}
<div className="legend-figure legend-druckenmiller"
     style={{backgroundImage: "url('/images/legends/stan-druckenmiller.jpg')"}}></div>

{/* Ben Graham */}
<div className="legend-figure legend-graham"
     style={{backgroundImage: "url('/images/legends/ben-graham.jpg')"}}></div>

{/* Peter Lynch */}
<div className="legend-figure legend-lynch"
     style={{backgroundImage: "url('/images/legends/peter-lynch.jpg')"}}></div>

{/* John Maynard Keynes */}
<div className="legend-figure legend-keynes"
     style={{backgroundImage: "url('/images/legends/john-keynes.jpg')"}}></div>

{/* David Tepper */}
<div className="legend-figure legend-tepper"
     style={{backgroundImage: "url('/images/legends/david-tepper.jpg')"}}></div>

{/* Terry Smith */}
<div className="legend-figure legend-smith"
     style={{backgroundImage: "url('/images/legends/terry-smith.jpg')"}}></div>
```

## Image Requirements

- **Format**: JPG or PNG
- **Size**: 400-600px width recommended (will be scaled)
- **Orientation**: Portrait (vertical) works best
- **Quality**: High resolution for crisp display
- **Style**: Professional photos, preferably speaking/presenting

## Design Notes

The images are designed to be **atmospheric** and **non-distracting**:
- **8% opacity**: Very subtle presence
- **Grayscale filter**: Maintains color consistency
- **Gradient fade**: Blends into background
- **Slight rotation**: Creates natural, candid feel
- **Strategic positioning**: Distributed across the layout

The effect is like having these legends subtly present, watching over and inspiring your investment decisions, without overwhelming the main content.

## Adjusting Visibility

If you want to make the images more or less visible, edit the opacity in `index.html`:

```css
.legend-figure {
  opacity: 0.08;  /* Change this value (0.05-0.15 recommended) */
  filter: grayscale(100%);
  /* ... */
}
```

## Tips for Best Results

1. Use photos where the person is **looking slightly away** from camera (not direct eye contact)
2. Choose images with **clean backgrounds** or easily separable subjects
3. Prefer **candid speaking/presenting moments** over posed portraits
4. Ensure **consistent lighting** across all images for cohesion
5. Consider using **black & white source images** for even more subtlety

Enjoy honoring these investing legends in your platform!
