# Bolded Quotes Feature

## Overview
This feature enhances the chat response by returning the full source documents with used quotes bolded, rather than just returning individual quote segments.

## How It Works

### 1. Request Flow
- User asks a question
- System retrieves relevant documents (with text as list of tuples)
- Documents are split into numbered sources `[1], [2], [3]...` for the LLM
- LLM responds with `source_numbers: [1, 3, 5]` indicating which quotes were used

### 2. Response Processing
- Sources are grouped by `text_id` (parent document)
- For each document:
  - All text segments are retrieved
  - Used segments are wrapped with `**text**` (markdown bold)
  - Full document text is reconstructed with highlights
  - Individual used quotes are listed separately

### 3. Response Schema

#### Old Schema (Before)
```json
{
  "main_text": "Answer to the question...",
  "sources": [
    {
      "slug": "torah-and-humility",
      "timestamp": "01:50:44,237-01:51:27,113",
      "text": "Individual quote text...",
      "text_id": "28152428-427c-49d8-952a-6f6db03c0a71"
    }
  ]
}
```

#### New Schema (After)
```json
{
  "main_text": "Answer to the question...",
  "sources": [
    {
      "slug": "torah-and-humility",
      "text_id": "28152428-427c-49d8-952a-6f6db03c0a71",
      "full_text": "Plain text here **bolded quote that was used** more plain text **another bolded quote**...",
      "used_quotes": [
        {
          "number": 5,
          "text": "bolded quote that was used",
          "timestamp": "01:50:44,237-01:51:27,113"
        },
        {
          "number": 6,
          "text": "another bolded quote",
          "timestamp": "01:51:27,113-01:52:01,047"
        }
      ],
      "timestamp_range": "01:50:44,237-01:52:01,047"
    }
  ]
}
```

## Key Benefits

1. **Context Preservation**: Users see the full document context, not just isolated quotes
2. **Visual Highlighting**: Used quotes are bolded with `**text**` for easy identification
3. **Better Understanding**: Users can see how quotes fit within the larger discourse
4. **Grouped by Document**: Multiple quotes from the same document are shown together

## Implementation Details

### Files Modified
1. `rag/app/schemas/response.py`
   - Added `UsedQuote` model for individual quotes
   - Updated `SourceItem` model with new structure

2. `rag/app/api/v1/chat.py`
   - Modified `full_response()` function
   - Added logic to group sources by document
   - Added logic to reconstruct full text with bolded segments

### Bold Formatting
- Uses markdown syntax: `**text**`
- Frontend can render this as HTML bold or keep as markdown
- Alternative formats can be used (HTML `<b>`, custom markers, etc.)

## Example Use Case

### User Question
"What does Rav say about important traits in man?"

### LLM Response
Returns source numbers: `[3, 6, 8, 10]`

### System Processing
1. Groups sources by document:
   - Document A: sources [3, 6]
   - Document B: sources [8, 10]

2. For Document A:
   - Retrieves all segments from original document
   - Marks segments 3 and 6 as used
   - Returns: "segment1 text **segment3 text (used)** segment4 text **segment6 text (used)**"

3. Frontend displays full context with visual highlighting

## Testing

To test the feature:
1. Make a chat request with `type_of_request: "full"`
2. Verify response includes `full_text` with `**bold**` markers
3. Verify `used_quotes` array contains all used quotes with metadata
4. Verify `timestamp_range` spans the full document

## Future Enhancements

Possible improvements:
- Add highlighting color customization
- Support multiple highlight styles for different quote types
- Add quote numbers in the full text (e.g., `**[5] quote text**`)
- Add "jump to quote" functionality in frontend
