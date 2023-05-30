CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A'�       creation_date         2019-11-12T17:25:01Z   data_specs_version        01.00.30   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Thttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.piControl.none.r1i1p1f1   grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      0Tue May 30 16:59:03 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Omon.hfds.gn.v20191112/hfds_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.yearmean.mul.areacello_piControl_v20191112.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-CM2_r1i1p1f1/hfds_ACCESS-CM2_r1i1p1f1_piControl.mergetime.nc
Thu Nov 03 22:04:00 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Omon.hfds.gn.v20191112/hfds_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.nc /Users/benjamin/Data/p22b/CMIP6/areacello/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Ofx.areacello.gn.v20191112/areacello_Ofx_ACCESS-CM2_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Omon.hfds.gn.v20191112/hfds_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.yearmean.mul.areacello_piControl_v20191112.fldsum.nc
2019-11-12T17:25:01Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: CM2-piControl; Local ID: bi889; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])    parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0001-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       hfds   variant_label         r1i1p1f1   version       	v20191112      cmor_version      3.4.0      tracking_id       1hdl:21.14100/c67fc64e-dbb3-4ec6-915b-dd86a771e6b6      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-12T17:24:58Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).             �1e+20).             �                Ab�   Ab��   Ab#��   Vj��Ab#�   Ab#��   Ab(|�   Ui�Ab(Ǩ   Ab(|�   Ab-^�   Vyg{Ab-��   Ab-^�   Ab2@�   V#?�Ab2��   Ab2@�   Ab7"�   V�_6Ab7m�   Ab7"�   Ab<�   �ЫAb<O�   Ab<�   Ab@�   ��/�AbA1�   Ab@�   AbEȠ   ��w�AbF�   AbEȠ   AbJ��   U��AbJ��   AbJ��   AbO��   ��PAbOר   AbO��   AbTn�   V�]�AbT��   AbTn�   AbYP�   ּ��AbY��   AbYP�   Ab^2�   ��bbAb^}�   Ab^2�   Abc�   V�iqAbc_�   Abc�   Abg��   WFFAbhA�   Abg��   Ablؠ   U�EBAbm#�   Ablؠ   Abq��   �O��Abr�   Abq��   Abv��   �@Abv�   Abv��   Ab{~�   V}�Ab{ɨ   Ab{~�   Ab�`�   ��eAb���   Ab�`�   Ab�B�   W<g�Ab���   Ab�B�   Ab�$�   V�hAb�o�   Ab�$�   Ab��   V�I�Ab�Q�   Ab��   Ab��   W��Ab�3�   Ab��   Ab�ʠ   WftAb��   Ab�ʠ   Ab���   V��Ab���   Ab���   Ab���   U�q]Ab�٨   Ab���   Ab�p�   V.�Ab���   Ab�p�   Ab�R�   ���Ab���   Ab�R�   Ab�4�   T�ԨAb��   Ab�4�   Ab��   V�F�Ab�a�   Ab��   Ab���   �眄Ab�C�   Ab���   Ab�ڠ   W*8Ab�%�   Ab�ڠ   Abļ�   U��Ab��   Abļ�   Abɞ�   V�Ab��   Abɞ�   Ab΀�   �*%ZAb�˨   Ab΀�   Ab�b�   ��Abӭ�   Ab�b�   Ab�D�   V��mAb؏�   Ab�D�   Ab�&�   W
p\Ab�q�   Ab�&�   Ab��   U��Ab�S�   Ab��   Ab��   �,4Ab�5�   Ab��   Ab�̠   V�neAb��   Ab�̠   Ab�   Uu�}Ab���   Ab�   Ab���   V��ZAb�ۨ   Ab���   Ab�r�   WMʇAb���   Ab�r�   Ab�T�   �
*Ab���   Ab�T�   Ac6�   V8��Ac��   Ac6�   Ac	�   W
]]Ac	c�   Ac	�   Ac��   V��~AcE�   Ac��   Acܠ   �^}?Ac'�   Acܠ   Ac��   U�
rAc	�   Ac��   Ac��   ս-Ac�   Ac��   Ac!��   W9ItAc!ͨ   Ac!��   Ac&d�   �`faAc&��   Ac&d�   Ac+F�   WO7Ac+��   Ac+F�   Ac0(�   ���Ac0s�   Ac0(�   Ac5
�   U�c�Ac5U�   Ac5
�   Ac9�   V!�zAc:7�   Ac9�   Ac>Π   ���Ac?�   Ac>Π   AcC��   WD�dAcC��   AcC��   AcH��   V2!�AcHݨ   AcH��   AcMt�   UhU�AcM��   AcMt�   AcRV�   V1ϿAcR��   AcRV�   AcW8�   ֞��AcW��   AcW8�   Ac\�   W(z�Ac\e�   Ac\�   Ac`��   ���'AcaG�   Ac`��   Aceޠ   V�ߘAcf)�   Aceޠ   Acj��   W7�zAck�   Acj��   Aco��   VlWiAco��   Aco��   Act��   W�ActϨ   Act��   Acyf�   VíjAcy��   Acyf�   Ac~H�   W1a�Ac~��   Ac~H�   Ac�*�   VL6SAc�u�   Ac�*�   Ac��   ��ԃAc�W�   Ac��   Ac��   V+��Ac�9�   Ac��   Ac�Р   W��Ac��   Ac�Р   Ac���   ֱeAc���   Ac���   Ac���   �B�Ac�ߨ   Ac���   Ac�v�   ֥^Ac���   Ac�v�   Ac�X�   U�#Ac���   Ac�X�   Ac�:�   ��koAc���   Ac�:�   Ac��   U�w�Ac�g�   Ac��   Ac���   ��41Ac�I�   Ac���   Ac��   �x�7Ac�+�   Ac��   Ac�    W�oAc��   Ac�    Ac¤�   V:�Ac��   Ac¤�   Acǆ�   ���Ac�Ѩ   Acǆ�   Ac�h�   V�$`Ac̳�   Ac�h�   Ac�J�   V���Acѕ�   Ac�J�   Ac�,�   ֨��Ac�w�   Ac�,�   Ac��   V�/�Ac�Y�   Ac��   Ac��   ֩�Ac�;�   Ac��   Ac�Ҡ   ��!Ac��   Ac�Ҡ   Ac鴠   V��PAc���   Ac鴠   Ac   Vז�Ac��   Ac   Ac�x�   ��0$Ac�è   Ac�x�   Ac�Z�   �h��Ac���   Ac�Z�   Ac�<�   ֍WAAc���   Ac�<�   Ad�   V_�Adi�   Ad�   Ad �   W5��AdK�   Ad �   Ad�   V��Ad-�   Ad�   AdĠ   U�nAd�   AdĠ   Ad��   �J\�Ad�   Ad��   Ad��   Wno�AdӨ   Ad��   Adj�   WP�AAd��   Adj�   Ad$L�   WB?Ad$��   Ad$L�   Ad).�   �u2TAd)y�   Ad).�   Ad.�   VB�0Ad.[�   Ad.�   Ad2�   VC��Ad3=�   Ad2�   Ad7Ԡ   U�́Ad8�   Ad7Ԡ   Ad<��   VFtlAd=�   Ad<��   AdA��   U���AdA�   AdA��   AdFz�   ��+AdFŨ   AdFz�   AdK\�   UEo�AdK��   AdK\�   AdP>�   T�!AdP��   AdP>�   AdU �   V���AdUk�   AdU �   AdZ�   VoO4AdZM�   AdZ�   Ad^�   VյaAd_/�   Ad^�   AdcƠ   V�$�Add�   AdcƠ   Adh��   ՙ�-Adh�   Adh��   Adm��   T��Admը   Adm��   Adrl�   WyƁAdr��   Adrl�   AdwN�   �;�OAdw��   AdwN�   Ad|0�   WF�*Ad|{�   Ad|0�   Ad��   R���Ad�]�   Ad��   Ad���   V꘦Ad�?�   Ad���   Ad�֠   V}�3Ad�!�   Ad�֠   Ad���   ֤�"Ad��   Ad���   Ad���   ֦�TAd��   Ad���   Ad�|�   V@,Ad�Ǩ   Ad�|�   Ad�^�   V�a�Ad���   Ad�^�   Ad�@�   V��Ad���   Ad�@�   Ad�"�   զk6Ad�m�   Ad�"�   Ad��   W'��Ad�O�   Ad��   Ad��   ՜ڼAd�1�   Ad��   Ad�Ƞ   �f��Ad��   Ad�Ƞ   Ad���   �v�IAd���   Ad���   Ad���   Wm�Ad�ר   Ad���   Ad�n�   W8�HAdŹ�   Ad�n�   Ad�P�   V�tAdʛ�   Ad�P�   Ad�2�   ֆt�Ad�}�   Ad�2�   Ad��   �mY#Ad�_�   Ad��   Ad���   V��Ad�A�   Ad���   Ad�ؠ   ՇG�Ad�#�   Ad�ؠ   Ad⺠   ��<=Ad��   Ad⺠   Ad眠   W�NAd��   Ad眠   Ad�~�   V�Ad�ɨ   Ad�~�   Ad�`�   ���AAd�   Ad�`�   Ad�B�   ��d�Ad���   Ad�B�   Ad�$�   V��Ad�o�   Ad�$�   Ae �   T.�JAe Q�   Ae �   Ae�   ��Ae3�   Ae�   Ae	ʠ   Vv�[Ae
�   Ae	ʠ   Ae��   W2��Ae��   Ae��   Ae��   W p�Ae٨   Ae��   Aep�   ��}1Ae��   Aep�   AeR�   ����Ae��   AeR�   Ae"4�   WI��Ae"�   Ae"4�   Ae'�   W�Ae'a�   Ae'�   Ae+��   ֦K�Ae,C�   Ae+��   Ae0ڠ   WDi�Ae1%�   Ae0ڠ   Ae5��   Wk�Ae6�   Ae5��   Ae:��   � b�Ae:�   Ae:��   Ae?��   U��Ae?˨   Ae?��   AeDb�   VE�AeD��   AeDb�   AeID�   V�CAeI��   AeID�   AeN&�   �r�AeNq�   AeN&�   AeS�   V��AeSS�   AeS�   AeW�   ��(�AeX5�   AeW�   Ae\̠   ե�4Ae]�   Ae\̠   Aea��   ֈ��Aea��   Aea��   Aef��   U��Aefۨ   Aef��   Aekr�   V�LAek��   Aekr�   AepT�   U�,Aep��   AepT�   Aeu6�   W�xAeu��   Aeu6�   Aez�   W!mKAezc�   Aez�   Ae~��   Vc�AeE�   Ae~��   Ae�ܠ   �oAe�'�   Ae�ܠ   Ae���   ֏_vAe�	�   Ae���   Ae���   ��3Ae��   Ae���   Ae���   V�C�Ae�ͨ   Ae���   Ae�d�   Wo�Ae���   Ae�d�   Ae�F�   V��qAe���   Ae�F�   Ae�(�   ���Ae�s�   Ae�(�   Ae�
�   W:XCAe�U�   Ae�
�   Ae��   ��ѮAe�7�   Ae��   Ae�Π   ՆF�Ae��   Ae�Π   Ae���   ��`Ae���   Ae���   Ae���   ��5�Ae�ݨ   Ae���   Ae�t�   UL܄Ae���   Ae�t�   Ae�V�   V9Aeá�   Ae�V�   Ae�8�   W"N�Aeȃ�   Ae�8�   Ae��   V���Ae�e�   Ae��   Ae���   ��pAe�G�   Ae���   Ae�ޠ   V�zUAe�)�   Ae�ޠ   Ae���   V��Ae��   Ae���   Aeࢠ   W/�Ae���   Aeࢠ   Ae儠   W=��Ae�Ϩ   Ae儠   Ae�f�   UGl2Ae걨   Ae�f�   Ae�H�   �;M'Ae   Ae�H�   Ae�*�   V5
Ae�u�   Ae�*�   Ae��   ���Ae�W�   Ae��   Ae��   ֫�Ae�9�   Ae��   AfР   V�"Af�   AfР   Af��   ���Af��   Af��   Af��   �}Afߨ   Af��   Afv�   ���Af��   Afv�   AfX�   Vu��Af��   AfX�   Af:�   V�eAf��   Af:�   Af �   ��Af g�   Af �   Af$��   V��Af%I�   Af$��   Af)�   ��^Af*+�   Af)�   Af.    V��Af/�   Af.    Af3��   V!dBAf3�   Af3��   Af8��   �e�Af8Ѩ   Af8��   Af=h�   V��Af=��   Af=h�   AfBJ�   �͔AfB��   AfBJ�   AfG,�   V"��AfGw�   AfG,�   AfL�   V��0AfLY�   AfL�   AfP�   V~�AfQ;�   AfP�   AfUҠ   �9mAfV�   AfUҠ   AfZ��   V��8AfZ��   AfZ��   Af_��   U��bAf_�   Af_��   Afdx�   �}f�Afdè   Afdx�   AfiZ�   S��BAfi��   AfiZ�   Afn<�   ָz�Afn��   Afn<�   Afs�   V�o�Afsi�   Afs�   Afx �   Ԩ�:AfxK�   Afx �   Af|�   W(�%Af}-�   Af|�   Af�Ġ   ��Af��   Af�Ġ   Af���   ֍�Af��   Af���   Af���   VP�Af�Ө   Af���   Af�j�   V�*kAf���   Af�j�   Af�L�   V��Af���   Af�L�   Af�.�   ���JAf�y�   Af�.�   Af��   WB�Af�[�   Af��   Af��   UwV�Af�=�   Af��   Af�Ԡ   ֿ��Af��   Af�Ԡ   Af���   V��BAf��   Af���   Af���   Wi��Af��   Af���   Af�z�   ��H�Af�Ũ   Af�z�   Af�\�   ���KAf���   Af�\�   Af�>�   W]�9Af���   Af�>�   Af� �   ���Af�k�   Af� �   Af��   ��0Af�M�   Af��   Af��   ֊*.Af�/�   Af��   Af�Ơ   V��UAf��   Af�Ơ   Af٨�   W��Af��   Af٨�   Afފ�   ��L�Af�ը   Afފ�   Af�l�   V��LAf㷨   Af�l�   Af�N�   �45�Af虨   Af�N�   Af�0�   �	�%Af�{�   Af�0�   Af��   V�J)Af�]�   Af��   Af���   �?�xAf�?�   Af���   Af�֠   ֕ggAf�!�   Af�֠   Ag ��   WU�Ag�   Ag ��   Ag��   U�)�Ag�   Ag��   Ag
|�   ֶ��Ag
Ǩ   Ag
|�   Ag^�   T��Ag��   Ag^�   Ag@�   VȶAg��   Ag@�   Ag"�   V��0Agm�   Ag"�   Ag�   �Y�AgO�   Ag�   Ag"�   V��)Ag#1�   Ag"�   Ag'Ƞ   W'~�Ag(�   Ag'Ƞ   Ag,��   ֋V�Ag,��   Ag,��   Ag1��   W]z�Ag1ר   Ag1��   Ag6n�   V��Ag6��   Ag6n�   Ag;P�   V?�Ag;��   Ag;P�   Ag@2�   V�#�Ag@}�   Ag@2�   AgE�   �C/�AgE_�   AgE�   AgI��   ֫(�AgJA�   AgI��   AgNؠ   UuIAgO#�   AgNؠ   AgS��   V��AgT�   AgS��   AgX��   ��AgX�   AgX��   Ag]~�   V���Ag]ɨ   Ag]~�   Agb`�   �EǥAgb��   Agb`�   AggB�   �6��Agg��   AggB�   Agl$�   Vh��Aglo�   Agl$�   Agq�   ���AgqQ�   Agq�   Agu�   W
|7Agv3�   Agu�   Agzʠ   U��"Ag{�   Agzʠ   Ag��   U�FrAg��   Ag��   Ag���   �`��Ag�٨   Ag���   Ag�p�   ֒x<Ag���   Ag�p�   Ag�R�   U�K�Ag���   Ag�R�   Ag�4�   VL=�Ag��   Ag�4�   Ag��   լk|Ag�a�   Ag��   Ag���   U�\4Ag�C�   Ag���   Ag�ڠ   V��Ag�%�   Ag�ڠ   Ag���   V |�Ag��   Ag���   Ag���   W�XAg��   Ag���   Ag���   Vg�TAg�˨   Ag���   Ag�b�   U�@UAg���   Ag�b�   Ag�D�   �>CAg���   Ag�D�   Ag�&�   ֖��Ag�q�   Ag�&�   Ag��   V��sAg�S�   Ag��   Ag��   �%�CAg�5�   Ag��   Ag�̠   T�!�Ag��   Ag�̠   AgҮ�   Vu��Ag���   AgҮ�   Agא�   ���Ag�ۨ   Agא�   Ag�r�   �S��Agܽ�   Ag�r�   Ag�T�   �sfAg៨   Ag�T�   Ag�6�   W6TAg恨   Ag�6�   Ag��   �q Ag�c�   Ag��   Ag���   W>PAg�E�   Ag���   Ag�ܠ   W&�\Ag�'�   Ag�ܠ   Ag���   V�j�Ag�	�   Ag���   Ag���   V�dAg��   Ag���   Ah��   Vߩ�Ahͨ   Ah��   Ahd�   ֤��Ah��   Ahd�   AhF�   V�UAh��   AhF�   Ah(�   WI<VAhs�   Ah(�   Ah
�   �{�AhU�   Ah
�   Ah�   Uʛ�Ah7�   Ah�   Ah Π   ֈe�Ah!�   Ah Π   Ah%��   Tٹ�Ah%��   Ah%��   Ah*��   �BL�Ah*ݨ   Ah*��   Ah/t�   ��̈́Ah/��   Ah/t�   Ah4V�   ���Ah4��   Ah4V�   Ah98�   W�Ah9��   Ah98�   Ah>�   ��iAh>e�   Ah>�   AhB��   Ւ��AhCG�   AhB��   AhGޠ   W&AhH)�   AhGޠ   AhL��   Ve�AhM�   AhL��   AhQ��   �FAhQ��   AhQ��   AhV��   ֝�AhVϨ   AhV��   Ah[f�   V�� Ah[��   Ah[f�   Ah`H�   WG�\Ah`��   Ah`H�   Ahe*�   V�jAheu�   Ahe*�   Ahj�   ֘p�AhjW�   Ahj�   Ahn�   V��Aho9�   Ahn�   AhsР   ����Aht�   AhsР   Ahx��   �S�6Ahx��   Ahx��   Ah}��   Sf��Ah}ߨ   Ah}��   Ah�v�   V��Ah���   Ah�v�   Ah�X�   ��}�Ah���   Ah�X�   Ah�:�   ֓�Ah���   Ah�:�   Ah��   �1[KAh�g�   Ah��   Ah���   WmmsAh�I�   Ah���   Ah��   ��Ah�+�   Ah��   Ah�    W��Ah��   Ah�    Ah���   Tm<�Ah��   Ah���   Ah���   V���Ah�Ѩ   Ah���   Ah�h�   �d��Ah���   Ah�h�   Ah�J�   V�Ah���   Ah�J�   Ah�,�   V�0Ah�w�   Ah�,�   Ah��   և�^Ah�Y�   Ah��   Ah��   ���Ah�;�   Ah��   Ah�Ҡ   V��MAh��   Ah�Ҡ   Ah˴�   V��Ah���   Ah˴�   AhЖ�   Vd�Ah��   AhЖ�   Ah�x�   � �[Ah�è   Ah�x�   Ah�Z�   �,�Ahڥ�   Ah�Z�   Ah�<�   �"/XAh߇�   Ah�<�   Ah��   ���Ah�i�   Ah��   Ah� �   V���Ah�K�   Ah� �   Ah��   V[�)Ah�-�   Ah��   Ah�Ġ   U۞$Ah��   Ah�Ġ   Ah���   VbAh��   Ah���   Ah���   V�E�Ah�Ө   Ah���   Aij�   V�MqAi��   Aij�   AiL�   �#cAi��   AiL�   Ai.�   VG�*Aiy�   Ai.�   Ai�   W6�NAi[�   Ai�   Ai�   V��Ai=�   Ai�   AiԠ   �ӈAi�   AiԠ   Ai��   V� �Ai�   Ai��   Ai#��   ջ"Ai#�   Ai#��   Ai(z�   V>CHAi(Ũ   Ai(z�   Ai-\�   Q�_YAi-��   Ai-\�   Ai2>�   WI�Ai2��   Ai2>�   Ai7 �   գ�Ai7k�   Ai7 �   Ai<�   ��z�Ai<M�   Ai<�   Ai@�   V.EAiA/�   Ai@�   AiEƠ   W�!�AiF�   AiEƠ   AiJ��   V"L�AiJ�   AiJ��   AiO��   �Pp5AiOը   AiO��   AiTl�   U�dAiT��   AiTl�   AiYN�   W=kAiY��   AiYN�   Ai^0�   V�\�Ai^{�   Ai^0�   Aic�   ��Aic]�   Aic�   Aig��   �~�.Aih?�   Aig��   Ail֠   V��DAim!�   Ail֠   Aiq��   ֆ=/Air�   Aiq��   Aiv��   U>��Aiv�   Aiv��   Ai{|�   V�|�Ai{Ǩ   Ai{|�   Ai�^�   U��dAi���   Ai�^�   Ai�@�   U;��Ai���   Ai�@�   Ai�"�   �z�Ai�m�   Ai�"�   Ai��   �!"�Ai�O�   Ai��   Ai��   V�{Ai�1�   Ai��   Ai�Ƞ   V��pAi��   Ai�Ƞ   Ai���   U���Ai���   Ai���   Ai���   U�Ai�ר   Ai���   Ai�n�   �"��Ai���   Ai�n�   Ai�P�   �X�Ai���   Ai�P�   Ai�2�   U�+�Ai�}�   Ai�2�   Ai��   UBIAi�_�   Ai��   Ai���   V�Ai�A�   Ai���   Ai�ؠ   Wp��Ai�#�   Ai�ؠ   Aiĺ�   T�;Ai��   Aiĺ�   Aiɜ�   ֱw2Ai��   Aiɜ�   Ai�~�   V�2�Ai�ɨ   Ai�~�   Ai�`�   ֌�Aiӫ�   Ai�`�   Ai�B�   ��/Ai؍�   Ai�B�   Ai�$�   �HAi�o�   Ai�$�   Ai��   Վ�Ai�Q�   Ai��   Ai��   W+qAi�3�   Ai��   Ai�ʠ   U}�Ai��   Ai�ʠ   Ai�   V-��Ai���   Ai�   Ai���   �N�MAi�٨   Ai���   Ai�p�   U��Ai���   Ai�p�   Ai�R�   W.\�Ai���   Ai�R�   Aj4�   WPUAj�   Aj4�   Aj	�   V���Aj	a�   Aj	�   Aj��   V��AjC�   Aj��   Ajڠ   �A|Aj%�   Ajڠ   Aj��   ��Aj�   Aj��   Aj��   W���Aj�   Aj��   Aj!��   ���vAj!˨   Aj!��   Aj&b�   V#+�Aj&��   Aj&b�   Aj+D�   �Z9wAj+��   Aj+D�   Aj0&�   V��uAj0q�   Aj0&�   Aj5�   V0v�Aj5S�   Aj5�   Aj9�   T�dAj:5�   Aj9�   Aj>̠   V��Aj?�   Aj>̠   AjC��   V���AjC��   AjC��   AjH��   �R[HAjHۨ   AjH��   AjMr�   �fIAjM��   AjMr�   AjRT�   T���AjR��   AjRT�   AjW6�   �ALAjW��   AjW6�   Aj\�   V�w:Aj\c�   Aj\�   Aj`��   ��s�AjaE�   Aj`��   Ajeܠ   V���Ajf'�   Ajeܠ   Ajj��   WC��Ajk	�   Ajj��   Ajo��   W4kAjo�   Ajo��   Ajt��   V��Ajtͨ   Ajt��   Ajyd�   V���Ajy��   Ajyd�   Aj~F�   U���Aj~��   Aj~F�   Aj�(�   W	��Aj�s�   Aj�(�   Aj�
�   �Ϝ�Aj�U�   Aj�
�   Aj��   է�Aj�7�   Aj��   Aj�Π   W
:hAj��   Aj�Π   Aj���   V��Aj���   Aj���   Aj���   V=u�Aj�ݨ   Aj���   Aj�t�   ���vAj���   Aj�t�   Aj�V�   �oA�Aj���   Aj�V�   Aj�8�   W$v�Aj���   Aj�8�   Aj��   ��[�Aj�e�   Aj��   Aj���   �AAj�G�   Aj���   Aj�ޠ   W8>�Aj�)�   Aj�ޠ   Aj���   �?"�Aj��   Aj���   Aj¢�   V��HAj���   Aj¢�   AjǄ�   U\8�Aj�Ϩ   AjǄ�   Aj�f�   ֢��Aj̱�   Aj�f�   Aj�H�   �7w�Ajѓ�   Aj�H�   Aj�*�   �Ԝ�Aj�u�   Aj�*�   Aj��   Ղi�Aj�W�   Aj��   Aj��   W�Aj�9�   Aj��   Aj�Р   V�(AAj��   Aj�Р   Aj鲠   ���Aj���   Aj鲠   Aj   �{�Aj�ߨ   Aj   Aj�v�   WasAj���   Aj�v�   Aj�X�   U5�Aj���   Aj�X�   Aj�:�   ����Aj���   Aj�:�   Ak�   W=lAkg�   Ak�   Ak��   �&�JAkI�   Ak��   Ak�   WǊAk+�   Ak�   Ak    �̼Ak�   Ak    Ak��   �P�Ak�   Ak��   Ak��   WAe�AkѨ   Ak��   Akh�   V�XPAk��   Akh�   Ak$J�   �FF&Ak$��   Ak$J�   Ak),�   V���Ak)w�   Ak),�   Ak.�   V!AUAk.Y�   Ak.�   Ak2�   U��iAk3;�   Ak2�   Ak7Ҡ   ��֯Ak8�   Ak7Ҡ   Ak<��   V˾:Ak<��   Ak<��   AkA��   V�,�AkA�   AkA��   AkFx�   V�vAkFè   AkFx�   AkKZ�   V5i�AkK��   AkKZ�   AkP<�   V��AkP��   AkP<�   AkU�   ֲ�GAkUi�   AkU�   AkZ �   �
f�AkZK�   AkZ �   Ak^�   ԍ��Ak_-�   Ak^�   AkcĠ   ֜��Akd�   AkcĠ   Akh��   V~�Akh�   Akh��   Akm��   WT�AkmӨ   Akm��   Akrj�   ����Akr��   Akrj�   AkwL�   �>�{Akw��   AkwL�   Ak|.�   V�-JAk|y�   Ak|.�   Ak��   �i�EAk�[�   Ak��   Ak��   V��Ak�=�   Ak��   Ak�Ԡ   V�'Ak��   Ak�Ԡ   Ak���   Vo�WAk��   Ak���   Ak���   �{�Ak��   Ak���   Ak�z�   V훖Ak�Ũ   Ak�z�   Ak�\�   ��c�Ak���   Ak�\�   Ak�>�   �v�[Ak���   Ak�>�   Ak� �   �֓