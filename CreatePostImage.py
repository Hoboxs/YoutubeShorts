import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def createPostImage(text, image_name, isThumbnail=False, max_width=1000, corner_radius=40):
    print("\n\n---------------------------------")
    print("Generating Intro Image")
    print("---------------------------------")

    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "AdditionalFiles" / "intro_logo.png"
    checkmark_path = base_dir / "AdditionalFiles" / "blue_checkmark.png"
    awards_folder = base_dir / "AdditionalFiles" / "Awards"

    # Validate required files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Intro logo not found: {image_path}")
    if not checkmark_path.exists():
        raise FileNotFoundError(f"Checkmark not found: {checkmark_path}")
    if not awards_folder.exists():
        raise FileNotFoundError(f"Awards folder not found: {awards_folder}")

    # Compact layout, less vertical padding
    padding_top = 20
    padding_bottom = 20
    padding_sides = 35
    border_width = 2
    image_padding = 8
    gap_between_sections = 15

    # --- Load fonts ---
    def load_font(path, size):
        try:
            return ImageFont.truetype(path, size=size)
        except IOError:
            print(f"  ⚠️  Font not found: {path} (using default)")
            return ImageFont.load_default()

    header_font = load_font(base_dir / "Fonts" / "Montserrat-Bold.ttf", 60)
    middle_font = load_font(base_dir / "Fonts" / "Montserrat-Regular.ttf", 45)
    footer_font = load_font(base_dir / "Fonts" / "Montserrat-Regular.ttf", 35)

    # --- Load assets ---
    logo = Image.open(image_path).convert("RGBA")
    checkmark = Image.open(checkmark_path).convert("RGBA")

    # --- Load award images ---
    award_files = [
        str(p)
        for p in awards_folder.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    award_images = random.sample(award_files, min(10, len(award_files)))
    award_icons = [Image.open(img).convert("RGBA") for img in award_images]

    # --- Measure header text ---
    temp_img = Image.new("RGBA", (max_width, 300))
    temp_draw = ImageDraw.Draw(temp_img)
    main_text = "@RedditStoryOwl"
    l, t, r, b = temp_draw.textbbox((0, 0), main_text, font=header_font)
    text_h = b - t
    text_w = r - l

    # --- Scale checkmark to match text height ---
    ck_w_orig, ck_h_orig = checkmark.size
    scale_ratio = text_h / ck_h_orig
    new_ck_w = int(ck_w_orig * scale_ratio)
    new_ck_h = int(ck_h_orig * scale_ratio)
    checkmark = checkmark.resize((new_ck_w, new_ck_h), Image.LANCZOS)

    # --- Awards sizing (larger now) ---
    awards_spacing = 8
    if award_icons:
        available_width = max_width - padding_sides * 2 - text_w - new_ck_w - image_padding * 2 - logo.width
        num_awards = len(award_icons)
        total_spacing = (num_awards - 1) * awards_spacing
        # make awards larger by adding a multiplier
        award_size = max(40, min(70, (available_width - total_spacing) // num_awards))
    else:
        award_size = 0

    # --- Combined height (text + awards) ---
    combined_block_height = text_h + (award_size + 10 if award_icons else 0)

    # --- Make logo bigger ---
    target_logo_height = max(100, int(combined_block_height * 1.7))  # 70% larger
    logo_ratio = logo.width / logo.height
    logo = logo.resize((int(target_logo_height * logo_ratio), target_logo_height), Image.LANCZOS)

    # --- Wrap middle text ---
    words = text.split(" ")
    lines, current = [], ""
    temp_draw2 = ImageDraw.Draw(Image.new("RGBA", (max_width, 1)))
    for w in words:
        test_line = f"{current} {w}".strip()
        l3, t3, r3, b3 = temp_draw2.textbbox((0, 0), test_line, font=middle_font)
        if (r3 - l3) <= (max_width - padding_sides * 2):
            current = test_line
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_height = temp_draw2.textbbox((0, 0), "A", font=middle_font)[3] - temp_draw2.textbbox((0, 0), "A", font=middle_font)[1] + 8
    text_height_middle = len(lines) * line_height

    # --- Dimensions ---
    header_height = max(logo.height, combined_block_height)
    footer_icon_size = 38
    footer_height = 70
    total_height = padding_top + header_height + gap_between_sections + text_height_middle + footer_height + padding_bottom


    img = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(border_width, border_width),
         (max_width - border_width, total_height - border_width)],
        radius=corner_radius,
        fill="white", outline="black", width=border_width
    )

    # --- HEADER ---
    logo_x = padding_sides
    logo_y = padding_top + (header_height - logo.height) // 2
    img.paste(logo, (logo_x, logo_y), logo)

    text_block_x = logo_x + logo.width + image_padding
    text_block_y = padding_top + (header_height - combined_block_height) // 2
    draw.text((text_block_x, text_block_y), main_text, font=header_font, fill="black")

    tb = draw.textbbox((text_block_x, text_block_y), main_text, font=header_font)
    text_width = tb[2] - tb[0]
    ck_x = text_block_x + text_width + 5
    ck_y = text_block_y + (text_h - new_ck_h) // 2
    img.paste(checkmark, (ck_x, ck_y), checkmark)

    # --- Awards row (now bigger) ---
    if award_icons:
        awards_y = text_block_y + text_h + 10
        awards_x = text_block_x
        for icon in award_icons:
            icon_resized = icon.resize((award_size, award_size), Image.LANCZOS)
            img.paste(icon_resized, (awards_x, awards_y), icon_resized)
            awards_x += award_size + awards_spacing

    # --- MIDDLE TEXT ---
    y = padding_top + header_height + gap_between_sections // 2
    for line in lines:
        draw.text((padding_sides, y), line, font=middle_font, fill="black")
        y += line_height

    # --- FOOTER ---
    footer_images = [
        str(base_dir / "AdditionalFiles" / "like.png"),
        str(base_dir / "AdditionalFiles" / "comment.png"),
        str(base_dir / "AdditionalFiles" / "share.png"),
    ]
    footer_texts = ["999+", "999+", "Share"]

    footer_y = total_height - footer_height - padding_bottom // 2
    left_x = padding_sides
    spacing_between = 25

    icon_like = Image.open(footer_images[0]).convert("RGBA").resize((footer_icon_size, footer_icon_size))
    img.paste(icon_like, (left_x, footer_y), icon_like)
    text_x = left_x + footer_icon_size + 8
    text_y = footer_y + (footer_icon_size - draw.textbbox((0, 0), footer_texts[0], font=footer_font)[3]) // 2
    draw.text((text_x, text_y), footer_texts[0], font=footer_font, fill="black")

    comment_x = text_x + draw.textbbox((0, 0), footer_texts[0], font=footer_font)[2] + spacing_between
    icon_comment = Image.open(footer_images[1]).convert("RGBA").resize((footer_icon_size, footer_icon_size))
    img.paste(icon_comment, (comment_x, footer_y), icon_comment)
    text_x2 = comment_x + footer_icon_size + 8
    text_y2 = footer_y + (footer_icon_size - draw.textbbox((0, 0), footer_texts[1], font=footer_font)[3]) // 2
    draw.text((text_x2, text_y2), footer_texts[1], font=footer_font, fill="black")

    icon_share = Image.open(footer_images[2]).convert("RGBA").resize((footer_icon_size, footer_icon_size))
    share_x = max_width - padding_sides - (footer_icon_size + 8 + draw.textbbox((0, 0), footer_texts[2], font=footer_font)[2])
    img.paste(icon_share, (share_x, footer_y), icon_share)
    text_x_share = share_x + footer_icon_size + 8
    text_y_share = footer_y + (footer_icon_size - draw.textbbox((0, 0), footer_texts[2], font=footer_font)[3]) // 2
    draw.text((text_x_share, text_y_share), footer_texts[2], font=footer_font, fill="black")

    # --- Thumbnail resize ---
    if isThumbnail:
        img = img.resize((1280, 720), Image.LANCZOS)

    img.save(image_name)
    # img.show()