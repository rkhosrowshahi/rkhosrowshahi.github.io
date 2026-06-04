---
layout: page
title: news
permalink: /news/
nav: true
nav_order: 9
---

{% assign news_items = site.news | sort: "date" | reverse %}

{% if news_items.size > 0 %}
{% for item in news_items %}
**{{ item.date | date: "%B %-d, %Y" }}**

{% if item.inline %}
{{ item.content }}
{% else %}
[{{ item.title }}]({{ item.url | relative_url }})
{% endif %}

{% unless forloop.last %}
***
{% endunless %}
{% endfor %}
{% else %}
No news yet.
{% endif %}
