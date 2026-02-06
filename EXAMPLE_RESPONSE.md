# Example Response - Bolded Quotes Feature

## Sample Question
"What does Rav say about important traits in man?"

## LLM Returns Source Numbers
```json
{
  "main_text": "Rav Soloveitchik emphasizes several important traits in man...",
  "source_numbers": [3, 6, 7, 8, 10]
}
```

## OLD Response Format (Before Implementation)
Each quote was returned separately without context:

```json
{
  "main_text": "Rav Soloveitchik emphasizes several important traits in man...",
  "sources": [
    {
      "slug": "concepts-of-jewish-education",
      "timestamp": "00:50:08,905-00:50:37,334",
      "text": "action leads to kedusha, disciplined social action leads to honesty, to emes, to chesed, to dignity.",
      "text_id": "b0174466-9983-444c-aaad-f77724337190"
    },
    {
      "slug": "torah-and-humility",
      "timestamp": "01:51:27,113-01:52:01,047",
      "text": "Small town boy made good. He felt he owed nothing. He owed nothing to anybody.",
      "text_id": "28152428-427c-49d8-952a-6f6db03c0a71"
    },
    {
      "slug": "torah-and-humility",
      "timestamp": "01:52:01,247-01:52:38,718",
      "text": "to his fellow man in a variety of ways. This principle prevails in the domain of material as well as in that of spiritual success and achievement.",
      "text_id": "28152428-427c-49d8-952a-6f6db03c0a71"
    }
    // ... more individual quotes
  ]
}
```

**Problems with old format:**
- Lost context - quotes appear isolated
- Hard to see relationship between consecutive quotes
- Same document split into multiple entries
- No way to see what came before/after the quote

## NEW Response Format (After Implementation)
Sources are grouped by document with full context and bolded used quotes:

```json
{
  "main_text": "Rav Soloveitchik emphasizes several important traits in man...",
  "sources": [
    {
      "slug": "concepts-of-jewish-education",
      "text_id": "b0174466-9983-444c-aaad-f77724337190",
      "full_text": "the area of carnal activities. Namely, with regard to human relations. There are tens or perhaps hundreds of precepts which are concerned not with the relations between man and his body, but between man and his neighbor, fellow man. Or what we call it in the vernacular, social morality. Or in our halakhic jargon, we call it mitzvos shebein adam l'chavero. Mitzvos which are related to the contact between man and his fellow man. While disciplined carnal **action leads to kedusha, disciplined social action leads to honesty, to emes, to chesed, to dignity. Or disciplined social action commands respect of the non-Jew. And I believe that the av zoken teaches the child** not only about shabbos, not only about treifos and kashrus, not only about sex morality, though all those things are very important. They are the cornerstones of our tradition. But he also teaches him about relationships between him and society. And those",
      "used_quotes": [
        {
          "number": 3,
          "text": "action leads to kedusha, disciplined social action leads to honesty, to emes, to chesed, to dignity. Or disciplined social action commands respect of the non-Jew. And I believe that the av zoken teaches the child",
          "timestamp": "00:50:08,905-00:50:37,334"
        }
      ],
      "timestamp_range": "00:49:13,016-00:51:00,624"
    },
    {
      "slug": "torah-and-humility",
      "text_id": "28152428-427c-49d8-952a-6f6db03c0a71",
      "full_text": "Shall I take my bread [inaudible] and my water and my flesh that I have killed for my shearers and give unto men of whom I know not whence they are? What does it express? The mentality of Naval who felt everything is his, self-made man. What would you call it in America? [inaudible] Self-made man. **Small town boy made good. He felt he owed nothing. He owed nothing to anybody. Meime velachmi tivchase [inaudible]. Such an approach is contrary to the very letter and spirit of Yahadus. The humble man must feel indebted** **to his fellow man in a variety of ways. This principle prevails in the domain of material as well as in that of spiritual success and achievement. Loyalty is a central gravity in Yahadus. It's very interesting, if we read the story of Avraham Avinu, Abraham's life, we don't understand one thing, namely, why is the Torah so**",
      "used_quotes": [
        {
          "number": 6,
          "text": "Small town boy made good. He felt he owed nothing. He owed nothing to anybody. Meime velachmi tivchase [inaudible]. Such an approach is contrary to the very letter and spirit of Yahadus. The humble man must feel indebted",
          "timestamp": "01:51:27,113-01:52:01,047"
        },
        {
          "number": 7,
          "text": "to his fellow man in a variety of ways. This principle prevails in the domain of material as well as in that of spiritual success and achievement. Loyalty is a central gravity in Yahadus. It's very interesting, if we read the story of Avraham Avinu, Abraham's life, we don't understand one thing, namely, why is the Torah so",
          "timestamp": "01:52:01,247-01:52:38,718"
        }
      ],
      "timestamp_range": "01:50:44,237-01:52:38,718"
    }
    // ... more documents
  ]
}
```

**Benefits of new format:**
✅ Full context preserved - see text before and after the quote
✅ Related quotes grouped together by document
✅ Visual highlighting with `**bold**` markers
✅ Individual quotes still available in `used_quotes` array
✅ Overall timestamp range for the full document
✅ Better understanding of the Rav's flow of thought

## Frontend Display Example

### How to Render

```javascript
// Assuming markdown renderer is available
sources.forEach(source => {
  // Display document header
  console.log(`Source: ${source.slug}`);
  console.log(`Time: ${source.timestamp_range}`);
  
  // Render full text with markdown bold
  // **text** will render as <b>text</b>
  const renderedText = markdownToHtml(source.full_text);
  console.log(renderedText);
  
  // Optionally show individual quotes list
  console.log(`Used ${source.used_quotes.length} quotes from this source:`);
  source.used_quotes.forEach(quote => {
    console.log(`  [${quote.number}] ${quote.timestamp}`);
  });
});
```

### Visual Result
```
Source: torah-and-humility
Time: 01:50:44,237-01:52:38,718

Shall I take my bread [inaudible] and my water... 
<b>Small town boy made good. He felt he owed nothing...</b>
<b>to his fellow man in a variety of ways...</b>

Used 2 quotes from this source:
  [6] 01:51:27,113-01:52:01,047
  [7] 01:52:01,247-01:52:38,718
```

## Migration Notes

If you have existing frontend code that expects the old schema:

### Before (Old Schema)
```javascript
response.sources.forEach(source => {
  console.log(source.text);  // Just the quote
  console.log(source.timestamp);  // Timestamp for this quote
});
```

### After (New Schema)
```javascript
response.sources.forEach(source => {
  // Full text with bolded quotes
  console.log(source.full_text);
  
  // Individual quotes if needed
  source.used_quotes.forEach(quote => {
    console.log(quote.text);  // Just the quote
    console.log(quote.timestamp);  // Timestamp for this quote
  });
  
  // Overall timestamp range
  console.log(source.timestamp_range);
});
```
