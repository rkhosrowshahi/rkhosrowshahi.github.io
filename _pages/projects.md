---
layout: page
title: projects
permalink: /projects/
description: Research projects in machine learning and optimization.
nav: true
nav_order: 4
display_categories: [research]
horizontal: false
---

<div class="projects">
{% assign sorted_projects = site.projects | sort: "importance" %}
<div class="row row-cols-1 row-cols-md-3">
  {% for project in sorted_projects %}
    {% include projects.liquid %}
  {% endfor %}
</div>
</div>
